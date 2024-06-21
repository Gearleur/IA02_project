import numpy as np

def state_to_hashable(state):
    return tuple(sorted((hex.q, hex.r, hex.s, occupant) for hex, occupant in state.items()))

def evaluate_state(state, player):
    if player == 1:
        return -np.sum(state == -1)
    else:
        return np.sum(state == 1)



def minimax_dodo(game, state, depth, alpha, beta, maximizingPlayer, player, memo = None):
    if memo is None:
        memo = {}
    
    hashable_state = state_to_hashable(state)
    if hashable_state in memo:
        return memo[hashable_state]
    
    if depth == 0 or game.is_terminal_state(state, player):
        evaluation = evaluate_state(game, player)
        memo[hashable_state] = (None, evaluation)
        return memo[hashable_state]

    valid_moves = game.get_valid_moves(state, player)

    if maximizingPlayer:
        maxEval = float('-inf')
        best_move = None
        for move in valid_moves:
            start, end = move
            next_state = game.get_next_state(state, start, end)
            currentEval = minimax_dodo(game, next_state, depth - 1, alpha, beta, False, -player, memo)[1]
            if currentEval > maxEval:
                maxEval = currentEval
                best_move = move
            alpha = max(alpha, currentEval)
            if beta <= alpha:
                break
        memo[hashable_state] = (best_move, maxEval)
        return memo[hashable_state]
    else:
        minEval = float('inf')
        best_move = None
        for move in valid_moves:
            start, end = move
            next_state = game.get_next_state(state, start, end)
            currentEval = minimax_dodo(game, next_state, depth - 1, alpha, beta, True, -player, memo)[1]
            if currentEval < minEval:
                minEval = currentEval
                best_move = move
            beta = min(beta, currentEval)
            if beta <= alpha:
                break
        memo[hashable_state] = (best_move, minEval)
        return memo[hashable_state]