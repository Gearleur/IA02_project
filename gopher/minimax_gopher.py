import numpy as np
from copy import deepcopy
from .hex import Hex


def state_to_hashable(state):
    return tuple(
        sorted((hex.q, hex.r, hex.s, occupant) for hex, occupant in state.items())
    )


def minimax_gopher(
    game, state, depth, alpha, beta, maximizingPlayer, player, memo=None
):
    if not state:
        return 0, Hex(0, -5, 5)

    if memo is None:
        memo = {}

    state_key = tuple(
        sorted((hex, val) for hex, val in state.items() if hex is not None)
    )
    if state_key in memo:
        return memo[state_key]

    if depth == 0:
        score = game.evaluate_state(state, player)
        memo[state_key] = (score, None)
        return score, None

    if game.is_terminal_state(state, player):
        score = game.evaluate_state_terminal(state, player)
        memo[state_key] = (score, None)
        return score, None

    valid_moves = game.get_valid_moves(state, player)
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
                break
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
                break
        memo[state_key] = (min_eval, best_move)
        return min_eval, best_move
