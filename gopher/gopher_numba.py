import numpy as np
from numba import jit, prange
from collections import namedtuple
from copy import deepcopy
import random
from collections import namedtuple
from typing import List, Tuple

Cell = Tuple[int, int]
Hex = namedtuple("Hex", ["q", "r", "s"])

def hex_neighbor(hex, direction):
    directions = [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)]
    dir = directions[direction]
    return (hex[0] + dir[0], hex[1] + dir[1], hex[2] + dir[2])

@jit(nopython=True)
def evaluate_state_numba(state, player):
    score = 0
    for i in range(state.shape[0]):
        if state[i, 3] == player:
            score += 1
        elif state[i, 3] == -player:
            score -= 1
    return score

@jit(nopython=True)
def evaluate_state_terminal_numba(state, player):
    return -1000

@jit(nopython=True, parallel=True)
def get_valid_moves_numba(size, state, player):
    valid_moves = []
    if state.shape[0] == 0:
        for q in prange(-size, size + 1):
            for r in prange(-size, size + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= size:
                    valid_moves.append((q, r, s))
    else:
        for q in prange(-size, size + 1):
            for r in prange(-size, size + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= size:
                    action = (q, r, s)
                    if is_valid_move_numba(state, action, player, size):
                        valid_moves.append(action)
    return valid_moves

@jit(nopython=True)
def is_valid_move_numba(state, action, player, size):
    if max(abs(action[0]), abs(action[1]), abs(action[2])) > size:
        return False

    for i in range(state.shape[0]):
        if (state[i, 0], state[i, 1], state[i, 2]) == action:
            return False

    has_enemy_connection = False
    has_friendly_connection = False

    for direction in range(6):
        neighbor = hex_neighbor(action, direction)
        if max(abs(neighbor[0]), abs(neighbor[1]), abs(neighbor[2])) <= size:
            for i in range(state.shape[0]):
                if (state[i, 0], state[i, 1], state[i, 2]) == (neighbor[0], neighbor[1], neighbor[2]):
                    if state[i, 3] == player:
                        has_friendly_connection = True
                    elif state[i, 3] != 0:
                        has_enemy_connection = True

    return not has_friendly_connection and has_enemy_connection

@jit(nopython=True, parallel=True)
def minimax_gopher_numba(state, depth, alpha, beta, maximizingPlayer, player, memo_keys, memo_vals, size):
    state_key = tuple(sorted([tuple(row) for row in state]))
    for i in range(len(memo_keys)):
        if memo_keys[i] == state_key:
            return memo_vals[i]

    if depth == 0:
        score = evaluate_state_numba(state, player)
        memo_keys.append(state_key)
        memo_vals.append((score, None))
        return score, None

    if len(get_valid_moves_numba(size, state, player)) == 0:
        score = evaluate_state_terminal_numba(state, player)
        memo_keys.append(state_key)
        memo_vals.append((score, None))
        return score, None

    valid_moves = get_valid_moves_numba(size, state, player)
    best_move = None

    if maximizingPlayer:
        max_eval = float("-inf")
        for move in range(len(valid_moves)):
            next_state = np.vstack((state, [valid_moves[move][0], valid_moves[move][1], valid_moves[move][2], player]))
            eval, _ = minimax_gopher_numba(
                next_state, depth - 1, alpha, beta, False, -player, memo_keys, memo_vals, size
            )
            if eval > max_eval:
                max_eval = eval
                best_move = valid_moves[move]
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        memo_keys.append(state_key)
        memo_vals.append((max_eval, best_move))
        return max_eval, best_move
    else:
        min_eval = float("inf")
        for move in range(len(valid_moves)):
            next_state = np.vstack((state, [valid_moves[move][0], valid_moves[move][1], valid_moves[move][2], player]))
            eval, _ = minimax_gopher_numba(
                next_state, depth - 1, alpha, beta, True, -player, memo_keys, memo_vals, size
            )
            if eval < min_eval:
                min_eval = eval
                best_move = valid_moves[move]
            beta = min(beta, eval)
            if beta <= alpha:
                break
        memo_keys.append(state_key)
        memo_vals.append((min_eval, best_move))
        return min_eval, best_move

class GopherGame3:
    def __init__(self, board_size=6):
        self.size = board_size - 1
        self.board_size = 2 * self.size + 1

    def __repr__(self):
        return "GopherGame"

    def get_initial_state(self):
        return np.empty((0, 4), dtype=np.int64)

    def get_current_player(self, state):
        return 1 if state.shape[0] % 2 == 0 else -1

    def get_next_state(self, state, action, player):
        return np.vstack((state, [action[0], action[1], action[2], player]))

    def is_valid_move(self, state, action, player):
        if max(abs(action[0]), abs(action[1]), abs(action[2])) > self.size:
            return False

        for i in range(state.shape[0]):
            if (state[i, 0], state[i, 1], state[i, 2]) == (action[0], action[1], action[2]):
                return False

        has_enemy_connection = False
        has_friendly_connection = False

        for direction in range(6):
            neighbor = hex_neighbor(action, direction)
            if max(abs(neighbor[0]), abs(neighbor[1]), abs(neighbor[2])) <= self.size:
                for i in range(state.shape[0]):
                    if (state[i, 0], state[i, 1], state[i, 2]) == (neighbor[0], neighbor[1], neighbor[2]):
                        if state[i, 3] == player:
                            has_friendly_connection = True
                        elif state[i, 3] != 0:
                            has_enemy_connection = True

        return not has_friendly_connection and has_enemy_connection

    def get_valid_moves(self, state, player):
        valid_moves = []
        size = self.size
        if state.shape[0] == 0:
            for q in range(-size, size + 1):
                for r in range(-size, size + 1):
                    s = -q - r
                    if max(abs(q), abs(r), abs(s)) <= size:
                        valid_moves.append((q, r, s))
        else:
            for q in range(-size, size + 1):
                for r in range(-size, size + 1):
                    s = -q - r
                    if max(abs(q), abs(r), abs(s)) <= size:
                        action = (q, r, s)
                        if self.is_valid_move(state, action, player):
                            valid_moves.append(action)
        return valid_moves

    def display(self, state):
        for r in range(-self.size, self.size + 1):
            indent = abs(r)
            print(" " * indent, end="")
            for q in range(-self.size, self.size + 1):
                s = -q - r
                if abs(q) <= self.size and abs(r) <= self.size and abs(s) <= self.size:
                    hex = (q, r, s)
                    found = False
                    for i in range(state.shape[0]):
                        if (state[i, 0], state[i, 1], state[i, 2]) == (hex[0], hex[1], hex[2]):
                            val = state[i, 3]
                            if val == 1:
                                print("R", end=" ")
                            elif val == -1:
                                print("B", end=" ")
                            found = True
                            break
                    if not found:
                        print(".", end=" ")
            print()

    def is_terminal_state(self, state, player):
        return len(self.get_valid_moves(state, player)) == 0

    def evaluate_state(self, state, player):
        score = 0
        for i in range(state.shape[0]):
            if state[i, 3] == player:
                score += 1
            elif state[i, 3] == -player:
                score -= 1
        return score
    
    def evaluate_state_terminal(self, state, player):
        return -1000

    def minimax_gopher(self, state, depth, alpha, beta, maximizingPlayer,memo_keys, memo_vals, player):
        return minimax_gopher_numba(state, depth, alpha, beta, maximizingPlayer, player, memo_keys, memo_vals, self.size)


