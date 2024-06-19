import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import logging

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
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlockGPU(num_hidden) for _ in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (2 * game.size + 1) ** 2, game.action_size) 
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * (2 * game.size + 1) ** 2, 1), 
            nn.Tanh()
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
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visite_count=0):
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
        return q_value + self.args['C'] * math.sqrt((self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0 :
                child_state = self.state.copy()
                child_state = self.game.get_next_state_encoded(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player = -1)
                
                child = NodeAlphaGPU(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
                logger.debug(f"Created new child node: {child_state} with action {action} and prior {prob}")
    
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
        self.device = model.module.device if isinstance(model, nn.DataParallel) else model.device
    
    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_states(states, 1), device=self.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
            
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves_encoded(states[i], 1)
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)
            
            spg.root = NodeAlphaGPU(self.game, self.args, states[i], visite_count=1)
            spg.root.expand(spg_policy)
        
        
        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root
                
                while node.is_fully_expanded():
                    node = node.select()
                    
                value, is_terminal = self.game.get_value_and_terminated(node.state, -1)
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node
                    
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
            
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_states(states, 1), device=self.device)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                value = value.cpu().numpy()
            
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]
                
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
        self.device = model.module.device if isinstance(model, nn.DataParallel) else model.device
        self.mcts = MCTSAlphaParallelGPU(game, args, model)
        
    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        
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
                
                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                
                spg.state = self.game.get_next_state_encoded(spg.state, action, player)
                
                value, is_terminal = self.game.get_value_and_terminated(spg.state, player)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state, -hist_player),
                            hist_action_probs,
                            hist_outcome
                        ))
                    
                    del spGames[i]
            
            player = self.game.get_opponent(player)
            
        return return_memory
            
            
    
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.device)
            
            # Ajoutez votre code de perte et d'optimisation ici
            
    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []
            
            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()
                
            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)
            
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")
            
            
class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None