from .hex_2 import Hex, Point, hex_add, hex_subtract, hex_neighbor, idx_to_hex, hex_to_idx
import random
import numpy as np
from typing import List, Tuple

directionB = [(1, 0), (1, -1), (0, -1)]
directionR = [(-1, 0), (-1, 1), (0, 1)]


class DodoGame:
    def __init__(self, board_size=4):
        self.size = board_size - 1
        self.current_player = 1
        self.action_size = board_size * board_size * board_size * board_size

    def __repr__(self):
        return "DodoGame"

    def get_initial_state(self):
        Position_bleu = [(0, 4), (0, 3), (0, 5), (0, 6), (1, 3), (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]
        Position_rouge = [(3, 0), (3, 1), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2), (5, 3), (6, 0),
                          (6, 1), (6, 2), (6, 3)]

        grid = np.zeros((2 * self.size + 1, 2 * self.size + 1), dtype=np.int8)
        for x, y in Position_bleu:
            grid[x, y] = -1

        for x, y in Position_rouge:
            grid[x, y] = 1
        return grid


    def get_next_state(self, state, action, pion, player):
        state[action[0], action[1]] = player
        state[pion[0], pion[1]] = 0
        return state
    
    def get_next_state_encoded(self, state, encoded_action, player):
        start, end = self.decode_action(encoded_action, 2 * self.size + 1)
        return self.get_next_state(state, end, start, player)


    def is_valid_move(self, grid, action, pion):

        if abs(action[0]) > 2 * self.size + 1 or abs(action[1]) > 2 * self.size + 1:
            return False

        if grid[action[0]][action[1]] != 0:
            return False

        if grid[pion[0]][pion[1]] != player:
            return False

        has_friendly_connection = False
        if player == 1:
            for d in directionR:
                if action[0] == d[0] + pion[0] and action[1] == d[1] + pion[1]:
                    has_friendly_connection = True
        if player == -1:
            for d in directionB:
                if action[0] == d[0] + pion[0] and action[1] == d[1] + pion[1]:
                    has_friendly_connection = True

        return has_friendly_connection

    def get_valid_moves(self, grid, player):
        valid_moves = []
        for pos_x in range(2 * self.size + 1):
            for pos_y in range(2 * self.size + 1):
                pos = [pos_x, pos_y]
                for act_x in range(2 * self.size + 1):
                    for act_y in range(2 * self.size + 1):
                        act = [act_x, act_y]
                        if self.is_valid_move(grid, act, pos, player):
                            valid_moves.append([pos, act])
        return valid_moves
    
    def get_valid_moves_encoded(self, grid, player=None):
        valid_moves = self.get_valid_moves(grid, player)
        encoded_moves = [self.encode_action(move, 2 * self.size + 1) for move in valid_moves]
        return encoded_moves

    def check_win(self, state, action, player):
        return len(self.get_valid_moves(state, player=player)) == 0

    def get_value_and_terminated(self, state, action, player):
        if self.check_win(state, action, player):
            return player, True
        return 0, False
    
    def get_encoded_state(self, state, player):
        board_size = 2 * self.size + 1
        current_player = player
        opponent = -current_player

        # Crée les couches pour l'état encodé
        layer_player = (state == current_player).astype(np.float32)
        layer_opponent = (state == opponent).astype(np.float32)
        layer_valid_moves = np.zeros_like(state, dtype=np.float32)

        valid_moves = self.get_valid_moves(state, player=current_player)
        for move in valid_moves:
            (start_x, start_y), (end_x, end_y) = move
            layer_valid_moves[end_x, end_y] = 1.0

        encoded_state = np.stack((layer_opponent, layer_valid_moves, layer_player), axis=0)
        return encoded_state
    
    def get_encoded_states(self, states, player):
        return [self.get_encoded_state(state, player) for state in states]
    
    def change_perspective(self, state, player):
        return player * state
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value

    def display(self, state):
        board_size = self.size
        for r in range(-board_size, board_size + 1):
            indent = abs(r)
            print(' ' * indent, end='')
            for q in range(-board_size, board_size + 1):
                s = -q - r
                if abs(q) <= board_size and abs(r) <= board_size and abs(s) <= board_size:
                    x, y = hex_to_idx(Hex(q, r, s), board_size)
                    if state[x][y] == 1:  # Remarquez que x et y sont inversés pour accéder correctement à state
                        print('R', end=' ')
                    elif state[x][y] == -1:
                        print('B', end=' ')
                    else:
                        print('.', end=' ')
            print()
            
    @staticmethod
    def encode_action(action, board_size):
        (start_x, start_y), (end_x, end_y) = action
        return (start_x * board_size + start_y) * board_size ** 2 + (end_x * board_size + end_y)

    @staticmethod
    def decode_action(encoded_action, board_size):
        total_positions = board_size ** 2
        start_pos = encoded_action // total_positions
        end_pos = encoded_action % total_positions

        start_x = start_pos // board_size
        start_y = start_pos % board_size

        end_x = end_pos // board_size
        end_y = end_pos % board_size

        return (start_x, start_y), (end_x, end_y)
