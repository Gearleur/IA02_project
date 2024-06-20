import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import logging
from torch.utils.data import DataLoader, TensorDataset
import os

from .hex import Hex, Point, hex_neighbor, hex_add, hex_subtract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResBlockGPU(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ResNetGPU(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()

        self.device = device
        game.size  # Taille du plateau

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
            [ResBlockGPU(num_hidden) for _ in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (2 * game.size + 1) ** 2, game.action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * (2 * game.size + 1) ** 2, 1),
            nn.Tanh(),
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class NodeAlphaGPU:
    def __init__(
        self, game, args, state, parent=None, action_taken=None, prior=0, visite_count=0
    ):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visite_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return (
            q_value
            + self.args["C"]
            * math.sqrt((self.visit_count) / (child.visit_count + 1))
            * child.prior
        )

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state_encoded(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = NodeAlphaGPU(
                    self.game, self.args, child_state, self, action, prob
                )
                self.children.append(child)

                logger.debug(
                    f"Created new child node: {child_state} with action {action} and prior {prob}"
                )

        return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTSAlphaParallelGPU:
    def __init__(self, game, args, model, player=1):
        self.game = game
        self.args = args
        self.model = model
        self.player = player
        self.device = (
            model.module.device if isinstance(model, nn.DataParallel) else model.device
        )
        self.corner_indices = [
            5,
            10,
            55,
            65,
            110,
            115,
        ]

    @torch.no_grad()
    def search(self, states, spGames):
        policies, _ = self.model(
            torch.tensor(self.game.get_encoded_states(states, 1), device=self.device)
        )
        policies = torch.softmax(policies, axis=1).cpu().numpy()
        policies = (1 - self.args["dirichlet_epsilon"]) * policies + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * self.game.action_size,
            size=policies.shape[0],
        )
        
        for i in range(policy.shape[0]):
            valid_moves = self.game.get_valid_moves_encoded(states[i], 1)
            for idx in self.corner_indices:
                if (
                    valid_moves[idx] == 1
                ):  # Vérifier si l'emplacement est un coup valide
                    policy[i, idx] *= self.args.get("corner_weight", 1.5)

        for i, spg in enumerate(spGames):
            spg_policy = policies[i]
            valid_moves = self.game.get_valid_moves_encoded(states[i], 1)
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = NodeAlphaGPU(self.game, self.args, states[i], visite_count=1)
            spg.root.expand(spg_policy)

        for search in range(self.args["num_searches"]):
            for spg in spGames:
                node = spg.root
                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, -1)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_spGames = [
                i for i, spg in enumerate(spGames) if spg.node is not None
            ]

            if expandable_spGames:
                states = np.stack([spGames[i].node.state for i in expandable_spGames])
                policies, values = self.model(
                    torch.tensor(
                        self.game.get_encoded_states(states, 1), device=self.device
                    )
                )
                policies = torch.softmax(policies, axis=1).cpu().numpy()
                values = values.cpu().numpy()

                for i, spg_idx in enumerate(expandable_spGames):
                    node = spGames[spg_idx].node
                    spg_policy, spg_value = policies[i], values[i]

                    valid_moves = self.game.get_valid_moves_encoded(node.state, 1)
                    spg_policy *= valid_moves
                    spg_policy /= np.sum(spg_policy)

                    node.expand(spg_policy)
                    node.backpropagate(spg_value)


class AlphaZeroParallelGPU:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.device = (
            model.module.device if isinstance(model, nn.DataParallel) else model.device
        )
        self.mcts = MCTSAlphaParallelGPU(game, args, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args["num_parallel_games"])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])

            neutral_states = self.game.change_perspective(states, player)
            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                if i == 0:
                    self.game.display(spg.state)
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (
                    1 / self.args["temperature"]
                )

                action = np.random.choice(
                    self.game.action_size, p=temperature_action_probs
                )

                spg.state = self.game.get_next_state_encoded(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(
                    spg.state, player
                )

                if is_terminal:
                    for (
                        hist_neutral_state,
                        hist_action_probs,
                        hist_player,
                    ) in spg.memory:
                        hist_outcome = (
                            value
                            if hist_player == player
                            else self.game.get_opponent_value(value)
                        )
                        return_memory.append(
                            (
                                self.game.get_encoded_state(
                                    hist_neutral_state, -hist_player
                                ),
                                hist_action_probs,
                                hist_outcome,
                            )
                        )

                    del spGames[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        dataset = TensorDataset(
            torch.tensor([m[0] for m in memory], dtype=torch.float32),
            torch.tensor([m[1] for m in memory], dtype=torch.float32),
            torch.tensor([m[2] for m in memory], dtype=torch.float32).reshape(-1, 1),
        )
        dataloader = DataLoader(
            dataset, batch_size=self.args["batch_size"], shuffle=True
        )

        for batch in dataloader:
            states, policy_targets, value_targets = batch
            states = states.to(self.device)
            policy_targets = policy_targets.to(self.device)
            value_targets = value_targets.to(self.device)

            self.optimizer.zero_grad()
            policy_preds, value_preds = self.model(states)
            loss_policy = F.cross_entropy(policy_preds, policy_targets)
            loss_value = F.mse_loss(value_preds, value_targets)
            loss = loss_policy + loss_value
            loss.backward()
            self.optimizer.step()

            # Ajoutez votre code de perte et d'optimisation ici

    def learn(self):
        # Spécifiez le répertoire de sauvegarde
        save_dir = "models"
        os.makedirs(save_dir, exist_ok=True)  # Crée le répertoire s'il n'existe pas
        
        for iteration in range(self.args["num_iterations"]):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(
                self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"]
            ):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args["num_epochs"]):
                self.train(memory)

            # Construire le chemin complet des fichiers de modèle et d'optimiseur
            model_path = os.path.join(save_dir, f"model_{iteration}_{self.game}.pt")
            optimizer_path = os.path.join(save_dir, f"optimizer_{iteration}_{self.game}.pt")
            
            # Enregistrer le modèle et l'état de l'optimiseur
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
