from .hex_2 import (
    Hex,
    Point,
    hex_add,
    hex_subtract,
    hex_neighbor,
    idx_to_hex,
    hex_to_idx,
)
import random
import numpy as np
from typing import List, Tuple

directionB = [(1, 0), (1, -1), (0, -1)]
directionR = [(-1, 0), (-1, 1), (0, 1)]


class DodoGame:
    def __init__(self):
        self.size = 3
        self.board_size = 2 * self.size + 1
        self.action_size = (
            self.board_size * self.board_size * self.board_size * self.board_size
        )

    def __repr__(self):
        return "DodoGame"

    def get_initial_state(self):
        Position_bleu = [
            (0, 4),
            (0, 3),
            (0, 5),
            (0, 6),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 5),
            (3, 6),
        ]
        Position_rouge = [
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (4, 2),
            (5, 0),
            (5, 1),
            (5, 2),
            (5, 3),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
        ]

        grid = np.zeros((2 * self.size + 1, 2 * self.size + 1), dtype=np.int8)
        for x, y in Position_bleu:
            grid[x, y] = -1

        for x, y in Position_rouge:
            grid[x, y] = 1
        return grid

    def get_next_state(self, state, action, player):
        (start_x, start_y), (end_x, end_y) = action
        state[end_x, end_y] = player
        state[start_x, start_y] = 0
        return state

    def get_next_state_encoded(self, state, encoded_action, player):
        start, end = self.decode_action(encoded_action)
        return self.get_next_state(state, (start, end), player)

    def is_valid_move(self, grid, action, player, change_perspective=False):
        (start_x, start_y), (end_x, end_y) = action
        q, r, s = idx_to_hex(end_x, end_y, self.size)

        if abs(q) > self.size or abs(r) > self.size or abs(s) > self.size:
            return False

        if grid[end_x][end_y] != 0:
            return False

        if grid[start_x][start_y] != player:
            return False

        has_friendly_connection = False
        if player == 1:
            for d in directionB if change_perspective else directionR:
                if end_x == d[0] + start_x and end_y == d[1] + start_y:
                    has_friendly_connection = True
        if player == -1:
            for d in directionB:
                if end_x == d[0] + start_x and end_y == d[1] + start_y:
                    has_friendly_connection = True

        return has_friendly_connection

    def get_valid_moves(self, grid, player, change_perspective=False):
        valid_moves = []
        for pos_x in range(2 * self.size + 1):
            for pos_y in range(2 * self.size + 1):
                start = (pos_x, pos_y)
                for act_x in range(2 * self.size + 1):
                    for act_y in range(2 * self.size + 1):
                        end = (act_x, act_y)
                        action = (start, end)
                        if self.is_valid_move(grid, action, player, change_perspective):
                            valid_moves.append(action)
        return valid_moves

    def get_valid_moves_encoded(self, grid, player, change_perspective=False):
        action_size = self.action_size
        valid_moves_encoded = np.zeros(action_size, dtype=np.int8)
        valid_moves = self.get_valid_moves(grid, player, change_perspective)
        for move in valid_moves:
            encoded_move = self.encode_action(move)
            valid_moves_encoded[encoded_move] = 1
        return valid_moves_encoded

    def check_win(self, state, player, change_perspective=False):
        return len(self.get_valid_moves(state, player, change_perspective)) == 0

    def get_value_and_terminated(self, state, player, change_perspective=False):
        if self.check_win(state, player, change_perspective):
            return player, True
        return 0, False

    def get_encoded_state(self, state, player, change_perspective=False):
        board_size = 2 * self.size + 1
        current_player = player
        opponent = -current_player

        # Crée les couches pour l'état encodé
        layer_player = (state == current_player).astype(np.float32)
        layer_opponent = (state == opponent).astype(np.float32)
        layer_valid_moves = np.zeros_like(state, dtype=np.float32)

        valid_moves = self.get_valid_moves(
            state, player=current_player, change_perspective=change_perspective
        )
        for move in valid_moves:
            (start_x, start_y), (end_x, end_y) = move
            layer_valid_moves[end_x, end_y] = 1.0

        encoded_state = np.stack(
            (layer_opponent, layer_valid_moves, layer_player), axis=0
        )
        return encoded_state

    def get_encoded_states(self, states, player, change_perspectives):
        encoded_states = []
        for state, change_perspective in zip(states, change_perspectives):
            encoded_state = self.get_encoded_state(state, player, change_perspective)
            encoded_states.append(encoded_state)
        return encoded_states

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
            print(" " * indent, end="")
            for q in range(-board_size, board_size + 1):
                s = -q - r
                if (
                    abs(q) <= board_size
                    and abs(r) <= board_size
                    and abs(s) <= board_size
                ):
                    x, y = hex_to_idx(Hex(q, r, s), board_size)
                    if (
                        state[x][y] == 1
                    ):  # Remarquez que x et y sont inversés pour accéder correctement à state
                        print("R", end=" ")
                    elif state[x][y] == -1:
                        print("B", end=" ")
                    else:
                        print(".", end=" ")
            print()

    def encode_action(self, action):
        (start_x, start_y), (end_x, end_y) = action
        return (start_x * (2 * self.size + 1) + start_y) * (2 * self.size + 1) ** 2 + (
            end_x * (2 * self.size + 1) + end_y
        )

    def decode_action(self, encoded_action):
        total_positions = (2 * self.size + 1) ** 2
        start_pos = encoded_action // total_positions
        end_pos = encoded_action % total_positions

        start_x = start_pos // (2 * self.size + 1)
        start_y = start_pos % (2 * self.size + 1)

        end_x = end_pos // (2 * self.size + 1)
        end_y = end_pos % (2 * self.size + 1)

        return (start_x, start_y), (end_x, end_y)
