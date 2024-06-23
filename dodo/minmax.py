import numpy as np


def state_to_hashable(state):
    return tuple(
        sorted((hex.q, hex.r, hex.s, occupant) for hex, occupant in state.items())
    )

#algorithme minmax avec élagage alpha-beta et mmemorisation.
def minimax_dodo(game, state, depth, alpha, beta, maximizingPlayer, player, memo=None):
    if memo is None:
        memo = {}
    # si l'état actuel est déjà mémorisé, la valeur associée est retournée
    hashable_state = (state_to_hashable(state), depth)
    if hashable_state in memo:
        return memo[hashable_state]
    # Si la profondeur est nulle, l'état est évalué, mémorisé et retourné
    if depth == 0:
        valuation = game.evaluate_state(state, player)
        memo[hashable_state] = (None, valuation)
        return memo[hashable_state]
    # Si l'état est terminal, il est évalué, mémorisé et retourné
    if game.is_terminal_state(state, player):
        valuation = game.evaluate_state(state, player)
        memo[hashable_state] = (None, valuation)
        return memo[hashable_state]
    # obtention des mouvements valides
    valid_moves = game.get_valid_moves(state, player)

    if maximizingPlayer:
        maxEval = float("-inf")
        best_move = valid_moves[
            0
        ]  # Initialiser avec un mouvement valide pour éviter None
        for move in valid_moves:
            start, end = move
            next_state = game.get_next_state(state, start, end, player)
            currentEval = minimax_dodo(
                game, next_state, depth - 1, alpha, beta, False, -player, memo
            )[1]
            if currentEval > maxEval:
                maxEval = currentEval
                best_move = move
            alpha = max(alpha, currentEval)
            if beta <= alpha:
                break
        memo[hashable_state] = (best_move, maxEval)
        return memo[hashable_state]
    else:
        minEval = float("inf")
        best_move = valid_moves[
            0
        ]  # Initialiser avec un mouvement valide pour éviter None
        for move in valid_moves:
            start, end = move
            next_state = game.get_next_state(state, start, end, player)
            currentEval = minimax_dodo(
                game, next_state, depth - 1, alpha, beta, True, -player, memo
            )[1]
            if currentEval < minEval:
                minEval = currentEval
                best_move = move
            beta = min(beta, currentEval)
            if beta <= alpha:
                break
        # mémorisation de l'état
        memo[hashable_state] = (best_move, minEval)
        return memo[hashable_state]
