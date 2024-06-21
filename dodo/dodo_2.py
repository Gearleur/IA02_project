from .hex_2 import (
    Hex,
    Point,
    hex_add,
    hex_subtract,
    idx_to_hex,
    hex_to_idx,
)

import random
import numpy as np
from typing import List, Tuple


class DodoGame2:
    def __init__(self):
        self.size = 3
        self.board_size = 2 * self.size + 1
        self.initial_state = self.init_board()
        self.memo = {}
        self.eval_cache = {}

    def init_board(self):
        state = {}
        for q in range(-self.size, self.size + 1):
            for r in range(-self.size, self.size + 1):
                s = -q - r
                if -self.size <= s <= self.size:
                    state[Hex(q, r, s)] = 0

        # Position des pions B (-1)
        positions_B = [(0, -3), (1, -3), (2, -3), (3, -3),
                       (0, -2), (1, -2), (2, -2), (3, -2),
                       (1, -1), (2, -1), (3, -1),
                       (2, 0), (3, 0)]
        for pos in positions_B:
            state[Hex(pos[0], pos[1], -pos[0]-pos[1])] = -1

        # Position des pions R (1)
        positions_R = [(0, 3), (-1, 3), (-2, 3), (-3, 3),
                       (0, 2), (-1, 2), (-2, 2), (-3, 2),
                       (-1, 1), (-2, 1), (-3, 1),
                       (-2, 0), (-3, 0)]
        for pos in positions_R:
            state[Hex(pos[0], pos[1], -pos[0]-pos[1])] = 1

        return state

    def hex_neighbors(self, hex, player):
        if player == 1:  # Rouge
            directions = [Hex(0, -1, 1), Hex(1, -1, 0), Hex(1, 0, -1)]
        elif player == -1:  # Bleu
            directions = [Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1)]
        neighbors = [hex_add(hex, direction) for direction in directions]
        return neighbors

    def get_valid_moves(self, state, player):
        valid_moves = []
        for hex, occupant in state.items():
            if occupant == player:
                neighbors = self.hex_neighbors(hex, player)
                for neighbor in neighbors:
                    if neighbor in state and state[neighbor] == 0:
                        valid_moves.append((hex, neighbor))
        return valid_moves

    def is_valid_move(self, state, start, end):
        return end in state and state[end] == 0

    def get_next_state(self, state, start, end, player):
        next_state = state.copy()
        next_state[end] = player
        next_state[start] = 0
        return next_state

    def is_terminal_state(self, state, player):
        return len(self.get_valid_moves(state, player)) == 0

    def display(self, state):
        board_size = self.size
        for r in range(-board_size, board_size + 1):
            indent = abs(r)
            print(" " * indent, end="")
            for q in range(-board_size, board_size + 1):
                s = -q - r
                if abs(q) <= board_size and abs(r) <= board_size and abs(s) <= board_size:
                    hex = Hex(q, r, s)
                    if hex in state:
                        val = state[hex]
                        if val == 1:
                            print("R", end=" ")
                        elif val == -1:
                            print("B", end=" ")
                        else:
                            print(".", end=" ")
            print()
            
    
    def evaluate_state_terminal(self, state, player):
        return 1000
        
        
    def evaluate_state(state, player):
        if player == 1:
            return -np.sum(state == -1)
        else:
            return np.sum(state == 1)