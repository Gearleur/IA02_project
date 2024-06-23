import numpy as np
from copy import deepcopy
from .hex import Hex
from typing import Dict, Tuple, Optional, List


def state_to_hashable(state: Dict[Hex, int]) -> Tuple:
    """
    Convertit l'état du jeu en une version hachable pour l'utilisation dans un dictionnaire.

    :param state: Dictionnaire représentant l'état du jeu avec les hexagones et leurs occupants.
    :return: Une version triée et hachable de l'état.
    """
    return tuple(
        sorted((hex.q, hex.r, hex.s, occupant) for hex, occupant in state.items())
    )

def minimax_gopher(
    game, state: Dict[Hex, int], depth: int, alpha: float, beta: float,
    maximizingPlayer: bool, player: int, memo: Optional[Dict] = None
) -> Tuple[int, Hex]:
    """
    Implémente l'algorithme Minimax avec élagage alpha-beta pour évaluer les meilleurs coups dans le jeu.

    :param game: Instance du jeu.
    :param state: État courant du jeu.
    :param depth: Profondeur maximale de recherche.
    :param alpha: Valeur alpha pour l'élagage.
    :param beta: Valeur beta pour l'élagage.
    :param maximizingPlayer: Booléen indiquant si le joueur actuel maximise le score.
    :param player: Joueur actuel (1 ou -1).
    :param memo: Dictionnaire pour mémoriser les états déjà évalués.
    :return: Tuple contenant le score évalué et le meilleur mouvement.
    """
    if not state:
        return 0, Hex(0, -5, 5)  # Retourner un score neutre si l'état est vide

    if memo is None:
        memo = {}

    # Convertir l'état en une clé hachable pour le mémo
    state_key = tuple(
        sorted((hex, val) for hex, val in state.items() if hex is not None)
    )
    if state_key in memo:
        return memo[state_key]  # Retourner le résultat mémorisé si déjà calculé

    if depth == 0:
        score = game.evaluate_state(state, player)  # Évaluer l'état pour le joueur actuel
        memo[state_key] = (score, None)
        return score, None

    if game.is_terminal_state(state, player):
        score = game.evaluate_state_terminal(state, player)  # Évaluer l'état terminal
        memo[state_key] = (score, None)
        return score, None

    valid_moves = game.get_valid_moves(state, player)  # Obtenir les coups valides
    best_move = None

    if maximizingPlayer:
        max_eval = float("-inf")
        for move in valid_moves:
            next_state = deepcopy(state)
            game.get_next_state(next_state, move, player)
            eval, _ = minimax_gopher(
                game, next_state, depth - 1, alpha, beta, False, -player, memo
            )
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Élaguer la branche
        memo[state_key] = (max_eval, best_move)
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in valid_moves:
            next_state = deepcopy(state)
            game.get_next_state(next_state, move, player)
            eval, _ = minimax_gopher(
                game, next_state, depth - 1, alpha, beta, True, -player, memo
            )
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Élaguer la branche
        memo[state_key] = (min_eval, best_move)
        return min_eval, best_move