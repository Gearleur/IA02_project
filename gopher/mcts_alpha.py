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


# Définition du bloc résiduel
class ResBlock(nn.Module):
    def __init__(self, num_hidden: int):
        """
        Initialise un bloc résiduel avec deux couches de convolution et de normalisation de lot.

        :param num_hidden: Nombre de canaux de sortie pour les convolutions et la normalisation.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Effectue une passe avant sur le bloc résiduel.

        :param x: Entrée du bloc résiduel.
        :return: Sortie du bloc résiduel après l'application de deux convolutions, normalisations et ReLU.
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


# Définition du réseau de neurones résiduel
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks: int, num_hidden: int, device: torch.device):
        """
        Initialise le réseau de neurones résiduel.

        :param game: Instance du jeu pour lequel le réseau est utilisé.
        :param num_resBlocks: Nombre de blocs résiduels.
        :param num_hidden: Nombre de canaux de sortie pour les convolutions.
        :param device: Appareil sur lequel exécuter le modèle (CPU ou GPU).
        """
        super().__init__()

        self.device = device
        self.board_size = game.size  # Taille du plateau de jeu

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Effectue une passe avant sur le réseau de neurones résiduel.

        :param x: Entrée du réseau (image de l'état du jeu).
        :return: Tuple contenant les sorties des têtes de politique et de valeur.
        """
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


# Définition de la classe de noeud pour l'algorithme MCTS Alpha
class NodeAlpha:
    def __init__(
        self,
        game,
        args: dict,
        state,
        parent=None,
        action_taken=None,
        prior: float = 0,
        visit_count: int = 0,
    ):
        """
        Initialise un noeud pour l'algorithme de recherche Monte Carlo Tree Search (MCTS).

        :param game: Instance du jeu pour lequel le noeud est utilisé.
        :param args: Dictionnaire des arguments de configuration.
        :param state: État du jeu représenté par ce noeud.
        :param parent: Noeud parent dans l'arbre de recherche.
        :param action_taken: Action prise pour atteindre cet état.
        :param prior: Probabilité a priori de choisir ce noeud.
        :param visit_count: Nombre de visites de ce noeud.
        """
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self) -> bool:
        """
        Vérifie si le noeud a été complètement étendu.

        :return: True si le noeud a des enfants, sinon False.
        """
        return len(self.children) > 0

    def select(self) -> "NodeAlpha":
        """
        Sélectionne le meilleur enfant basé sur la valeur UCB (Upper Confidence Bound).

        :return: Le meilleur noeud enfant.
        """
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child: "NodeAlpha") -> float:
        """
        Calcule la valeur UCB pour un enfant donné.

        :param child: Noeud enfant pour lequel calculer l'UCB.
        :return: Valeur UCB du noeud enfant.
        """
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

    def expand(self, policy: np.ndarray) -> "NodeAlpha":
        """
        Étend le noeud en ajoutant des enfants pour chaque action possible basée sur la politique.

        :param policy: Politique fournissant les probabilités pour chaque action.
        :return: Le dernier enfant ajouté.
        """
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state_encoded(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = NodeAlpha(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

                logger.debug(
                    f"Created new child node: {child_state} with action {action} and prior {prob}"
                )

        return child

    def backpropagate(self, value: float):
        """
        Propagation en arrière des valeurs de retour pour mettre à jour les valeurs et le compte de visites.

        :param value: Valeur de retour à propager.
        """
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTSAlpha:
    def __init__(self, game, args: dict, model: torch.nn.Module) -> None:
        """
        Initialise l'instance de MCTSAlpha.

        :param game: L'instance du jeu.
        :param args: Les arguments pour MCTS.
        :param model: Le modèle de réseau de neurones.
        """
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state) -> np.ndarray:
        """
        Effectue une recherche Monte Carlo Tree Search sur l'état donné.

        :param state: L'état initial du jeu.
        :return: Les probabilités des actions à prendre.
        """
        root = NodeAlpha(self.game, self.args, state, visite_count=1)

        # Obtenez la politique initiale du modèle
        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state(state, 1), device=self.model.device
            ).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

        # Ajouter du bruit de Dirichlet pour exploration
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size)

        # Filtrer les mouvements invalides
        valid_moves = self.game.get_valid_moves_encoded(state, 1)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        # Recherche
        for _ in range(self.args["num_searches"]):
            node = root

            # Descente dans l'arbre
            while node.is_fully_expanded():
                node = node.select()

            # Obtenez la valeur et l'état terminal du noeud
            value, is_terminal = self.game.get_value_and_terminated(node.state, -1)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                # Obtenez la politique et la valeur du modèle
                policy, value = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(node.state, 1),
                        device=self.model.device,
                    ).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

                # Filtrer les mouvements invalides
                valid_moves = self.game.get_valid_moves_encoded(node.state, 1)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()
                node.expand(policy)

            node.backpropagate(value)

        # Calculer les probabilités d'actions finales
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs


class AlphaZero:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, game, args: dict) -> None:
        """
        Initialise l'instance de AlphaZero.

        :param model: Le modèle de réseau de neurones.
        :param optimizer: L'optimiseur pour entraîner le modèle.
        :param game: L'instance du jeu.
        :param args: Les arguments pour AlphaZero.
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSAlpha(game, args, model)

    def selfPlay(self) -> list:
        """
        Effectue une partie contre soi-même pour générer des données d'entraînement.

        :return: La mémoire contenant les états, les probabilités d'actions et les résultats.
        """
        memory = []
        player = 1
        state = self.game.get_initial_state()  # Obtenir l'état initial du jeu

        while True:
            # Changer la perspective de l'état pour le joueur actuel
            neutral_state = self.game.change_perspective(state, player)
            # Effectuer une recherche MCTS pour obtenir les probabilités d'action
            action_probs = self.mcts.search(neutral_state)

            # Ajouter l'état, les probabilités d'action et le joueur actuel à la mémoire
            memory.append((neutral_state, action_probs, player))

            # Appliquer la température aux probabilités d'action pour favoriser l'exploration
            temperature_action_probs = action_probs ** (1 / self.args["temperature"])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            # Mettre à jour l'état du jeu en fonction de l'action choisie
            state = self.game.get_next_state_encoded(state, action, player)

            # Obtenir la valeur de l'état et vérifier si le jeu est terminé
            value, is_terminal = self.game.get_value_and_terminated(state, player)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    # Calculer le résultat de l'état historique en fonction du joueur
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append(
                        (self.game.get_encoded_state(hist_neutral_state, 1), hist_action_probs, hist_outcome)
                    )
                return returnMemory  # Retourner la mémoire des états joués

            # Passer au joueur suivant
            player = self.game.get_opponent(player)

    def train(self, memory: list) -> None:
        """
        Entraîne le modèle avec les données de la mémoire.

        :param memory: La mémoire contenant les états, les probabilités d'actions et les résultats.
        """
        random.shuffle(memory)  # Mélanger la mémoire pour l'entraînement
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx: min(len(memory), batchIdx + self.args["batch_size"])]
            state, policy_targets, value_targets = zip(*sample)

            # Convertir les échantillons en tenseurs PyTorch
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)

            # Passer les états à travers le modèle pour obtenir les prédictions
            out_policy, out_value = self.model(state)

            # Calculer la perte de la politique et la perte de valeur
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # Rétropropagation et optimisation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self) -> None:
        """
        Entraîne le modèle sur plusieurs itérations en utilisant self-play et des mises à jour de poids.
        """
        for iteration in range(self.args["num_iterations"]):
            memory = []

            # Met le modèle en mode évaluation
            self.model.eval()
            for _ in trange(self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"]):
                memory += self.selfPlay()

            # Met le modèle en mode entraînement
            self.model.train()
            for _ in trange(self.args["num_epochs"]):
                self.train(memory)

            # Sauvegarder l'état du modèle et de l'optimiseur
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")


class MCTSAlphaParallel:
    def __init__(self, game, args, model, player=1):
        """
        Initialisation de la classe MCTSAlphaParallel.

        :param game: Instance du jeu.
        :param args: Arguments pour la configuration.
        :param model: Modèle de réseau de neurones utilisé.
        :param player: Joueur courant (1 par défaut).
        """
        self.game = game
        self.args = args
        self.model = model
        self.player = player
        self.corner_indices = [
            5, 10, 55, 65, 110, 115,
        ]  # Indices des coins du plateau

    @torch.no_grad()
    def search(self, states: np.ndarray, spGames: list) -> None:
        """
        Effectue une recherche MCTS pour les états donnés.

        :param states: États actuels du jeu.
        :param spGames: Liste des jeux en parallèle.
        """
        # Obtenir les politiques et les valeurs prédites par le modèle pour les états donnés
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_states(states, 1), device=self.model.device)
        )
        # Appliquer la softmax pour obtenir des probabilités
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        # Mélanger les politiques avec du bruit Dirichlet pour ajouter de l'exploration
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * self.game.action_size, size=policy.shape[0]
        )

        # Ajuster les probabilités pour favoriser les coups valides dans les coins
        for i in range(policy.shape[0]):
            valid_moves = self.game.get_valid_moves_encoded(states[i], 1)
            for idx in self.corner_indices:
                if valid_moves[idx] == 1:  # Vérifier si l'emplacement est un coup valide
                    policy[i, idx] *= self.args.get("corner_weight", 1.5)

        # Élargir chaque jeu parallèle avec les politiques ajustées
        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves_encoded(states[i], 1)
            spg_policy *= valid_moves  # Masquer les coups invalides
            spg_policy /= np.sum(spg_policy)  # Normaliser les probabilités

            spg.root = NodeAlpha(self.game, self.args, states[i], visite_count=1)
            spg.root.expand(spg_policy)  # Élargir le noeud racine avec la politique

        # Effectuer les recherches MCTS
        for _ in range(self.args["num_searches"]):
            for spg in spGames:
                spg.node = None
                node = spg.root

                # Sélectionner jusqu'à atteindre un noeud non entièrement élargi
                while node.is_fully_expanded():
                    node = node.select()

                # Obtenir la valeur et l'état terminal du noeud
                value, is_terminal = self.game.get_value_and_terminated(node.state, -1)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)  # Rétropropagation de la valeur
                else:
                    spg.node = node  # Mettre à jour le noeud courant

            # Rassembler les jeux parallèles extensibles
            expandable_spGames = [
                idx for idx, spg in enumerate(spGames) if spg.node is not None
            ]

            if expandable_spGames:
                # Obtenir les états des noeuds extensibles
                states = np.stack(
                    [spGames[idx].node.state for idx in expandable_spGames]
                )

                # Obtenir les politiques et les valeurs pour les états extensibles
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_states(states, 1), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

            # Élargir et rétropropager pour chaque jeu parallèle extensible
            for i, idx in enumerate(expandable_spGames):
                node = spGames[idx].node
                spg_policy, spg_value = policy[i], value[i]

                valid_moves = self.game.get_valid_moves_encoded(node.state, 1)
                spg_policy *= valid_moves  # Masquer les coups invalides
                spg_policy /= np.sum(spg_policy)  # Normaliser les probabilités

                node.expand(spg_policy)  # Élargir le noeud
                node.backpropagate(spg_value)  # Rétropropagation de la valeur


class AlphaZeroParallel:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, game, args: dict):
        """
        Initialisation de la classe AlphaZeroParallel.

        :param model: Modèle de réseau de neurones utilisé.
        :param optimizer: Optimiseur pour le modèle.
        :param game: Instance du jeu.
        :param args: Arguments pour la configuration.
        """
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSAlphaParallel(game, args, model)

    def selfPlay(self) -> list:
        """
        Effectue une session de self-play en parallèle.

        :return: Mémoire des états joués.
        """
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for _ in range(self.args["num_parallel_games"])]

        while spGames:
            # Obtenir les états actuels de tous les jeux en parallèle
            states = np.stack([spg.state for spg in spGames])
            # Changer la perspective des états pour le joueur actuel
            neutral_states = self.game.change_perspective(states, player)
            # Effectuer la recherche MCTS sur les états actuels
            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]
                if i == 0:
                    self.game.display(spg.state)
                
                # Initialiser les probabilités d'action à zéro
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                # Ajouter l'état actuel, les probabilités d'action et le joueur à la mémoire
                spg.memory.append((spg.root.state, action_probs, player))

                # Appliquer la température aux probabilités d'action
                temperature_action_probs = action_probs ** (1 / self.args["temperature"])
                temperature_action_probs /= np.sum(temperature_action_probs)

                # Sélectionner une action en fonction des probabilités
                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                # Mettre à jour l'état du jeu en fonction de l'action choisie
                spg.state = self.game.get_next_state_encoded(spg.state, action, player)

                # Obtenir la valeur de l'état et vérifier si le jeu est terminé
                value, is_terminal = self.game.get_value_and_terminated(spg.state, player)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        # Calculer le résultat de l'état historique en fonction du joueur
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                        return_memory.append(
                            (self.game.get_encoded_state(hist_neutral_state, 1), hist_action_probs, hist_outcome)
                        )
                    # Supprimer le jeu terminé de la liste des jeux en parallèle
                    del spGames[i]

            # Passer au joueur suivant
            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory: list) -> None:
        """
        Entraîne le modèle avec la mémoire des états joués.

        :param memory: Mémoire des états joués.
        """
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx: min(len(memory), batchIdx + self.args["batch_size"])]
            state, policy_targets, value_targets = zip(*sample)

            # Convertir les échantillons en tenseurs PyTorch
            state = torch.tensor(np.array(state), dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(np.array(policy_targets), dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(np.array(value_targets).reshape(-1, 1), dtype=torch.float32, device=self.model.device)

            # Passer les états à travers le modèle pour obtenir les prédictions
            out_policy, out_value = self.model(state)

            # Calculer la perte de la politique et la perte de valeur
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            # Rétropropagation et optimisation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self) -> None:
        """
        Effectue le processus d'apprentissage complet sur plusieurs itérations.
        """
        for iteration in range(self.args["num_iterations"]):
            memory = []

            # Met le modèle en mode évaluation
            self.model.eval()
            for _ in trange(self.args["num_selfPlay_iterations"] // self.args["num_parallel_games"]):
                memory += self.selfPlay()

            # Met le modèle en mode entraînement
            self.model.train()
            for _ in trange(self.args["num_epochs"]):
                self.train(memory)

            # Sauvegarde l'état du modèle et de l'optimiseur
            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")
            
            
class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None


class MockResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super(MockResNet, self).__init__()
        self.device = device
        self.action_size = game.action_size
        self.num_hidden = num_hidden
        self.to(device)

    def forward(self, x):
        batch_size = x.size(0)
        # Politique factice : uniformément répartie
        policy = (
            torch.ones((batch_size, self.action_size), device=x.device)
            / self.action_size
        )
        # Valeur factice : nulle
        value = torch.zeros((batch_size, 1), device=x.device)
        return policy, value
