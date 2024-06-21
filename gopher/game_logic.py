# gopher_game/game/game_logic.py
from .hex import Hex, Point, hex_add, hex_subtract, hex_neighbor, idx_to_hex, hex_to_idx
import random
import numpy as np
from typing import List, Tuple

Cell = Tuple[int, int]


class GopherGame:
    def __init__(self, board_size=6):
        self.size = board_size - 1
        self.current_player = 1
        self.action_size = (2 * self.size + 1) ** 2

    def __repr__(self):
        return "GopherGame"

    def get_initial_state(self):
        return np.zeros((2 * self.size + 1, 2 * self.size + 1), dtype=np.int8)

    def get_current_player(self, state):
        return 1 if np.sum(state) % 2 == 0 else -1

    def get_next_state(self, state, action, player):
        q, r, s = action
        row = r + self.size
        col = q + self.size
        state[row, col] = player
        return state

    def get_next_state_idx(self, state, action, player):
        row, col = action
        state[row, col] = player
        return state

    def get_next_state_encoded(self, state, action, player):
        rows, cols = 2 * self.size + 1, 2 * self.size + 1
        row, col = np.unravel_index(action, (rows, cols))
        state[row, col] = player
        return state

    def is_valid_move(self, state, action, player=None):
        if player is None:
            player = self.get_current_player(state)

        # Vérifiez si le mouvement est à l'intérieur des limites de l'hexagone
        if (
            abs(action.q) > self.size
            or abs(action.r) > self.size
            or abs(action.s) > self.size
        ):
            return False

        # Vérifiez si la cellule est déjà occupée
        x, y = hex_to_idx(action, self.size)
        if state[x][y] != 0:  # Assumant que 0 signifie une cellule non occupée
            return False

        has_enemy_connection = False
        has_friendly_connection = False

        for direction in range(6):
            neighbor = hex_neighbor(action, direction)
            if (
                abs(neighbor.q) <= self.size
                and abs(neighbor.r) <= self.size
                and abs(neighbor.s) <= self.size
            ):
                nx, ny = hex_to_idx(neighbor, self.size)
                if state[nx][ny] == player:
                    has_friendly_connection = True
                elif state[nx][ny] != 0:
                    has_enemy_connection = True

        return not has_friendly_connection and has_enemy_connection

    def get_valid_moves(self, state, player=None):
        valid_moves = set()
        size = self.size
        is_empty = not any(state.flatten())  # Check if the state matrix is empty
        if player is None:
            current_player = self.get_current_player(state)
        else:
            current_player = player

        if is_empty:
            for q in range(-size, size + 1):
                for r in range(-size, size + 1):
                    s = -q - r
                    x, y = hex_to_idx(Hex(q, r, s), size)
                    if abs(q) <= size and abs(r) <= size and abs(s) <= size:
                        valid_moves.add((x, y))
        else:
            for q in range(-size, size + 1):
                for r in range(-size, size + 1):
                    s = -q - r
                    action = Hex(q, r, s)
                    x, y = hex_to_idx(action, size)
                    if self.is_valid_move(state, action, current_player):
                        x, y = hex_to_idx(action, size)
                        valid_moves.add((x, y))
        return list(valid_moves)

    def get_valid_moves_encoded(self, state, player=None):
        if player is None:
            player = self.get_current_player(state)

        board_size = self.size
        valid_moves_encoded = np.zeros(
            (2 * board_size + 1, 2 * board_size + 1), dtype=np.float32
        )
        valid_moves = self.get_valid_moves(state, player)

        for x, y in valid_moves:
            valid_moves_encoded[x, y] = 1

        return valid_moves_encoded.flatten()

    def check_win(self, state, player):
        return len(self.get_valid_moves(state, player=-player)) == 0

    def get_value_and_terminated(self, state, player=None):
        if player is None:
            player = self.get_current_player(state)

        if self.check_win(state, player):
            return 1, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return player * state

    def get_encoded_state(self, state, player=None):
        if player is None:
            player = self.get_current_player(state)

        board_size = self.size
        current_player = player
        opponent = -current_player

        # Crée les couches pour l'état encodé
        layer_player = (state == current_player).astype(np.float32)
        layer_opponent = (state == opponent).astype(np.float32)
        layer_valid_moves = np.zeros(
            (2 * board_size + 1, 2 * board_size + 1), dtype=np.float32
        )

        valid_moves = self.get_valid_moves(state, player=player)
        for x, y in valid_moves:
            layer_valid_moves[x, y] = 1

        # Empile les couches
        encoded_state = np.stack(
            (layer_opponent, layer_valid_moves, layer_player), axis=0
        )

        return encoded_state

    def get_encoded_states(self, states, player=None):
        if player is None:
            player = self.get_current_player(states[0])
        encoded_states = [self.get_encoded_state(state, player) for state in states]
        return np.array(encoded_states)

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

    def serveur_state_to_gopher(
        self, server_state: List[Tuple[Tuple[int, int], int]]
    ) -> List[List[int]]:
        # Taille du tableau
        array_size = 2 * self.size + 1

        # Initialiser le tableau 2D avec 0
        board = np.zeros((array_size, array_size), dtype=np.int8)

        # Utiliser cell_to_grid pour convertir et remplir le tableau
        for cell, value in server_state:
            q = cell[0]
            r = cell[1]
            s = -q + r
            hex = Hex(q, -r, s)
            x, y = hex_to_idx(hex, self.size)
            if value == 1:
                board[x][y] = 1
            elif value == 2:
                board[x][y] = -1

        return board

    def encoded_to_server(self, encoded):
        rows, cols = 2 * self.size + 1, 2 * self.size + 1
        row, col = np.unravel_index(encoded, (rows, cols))
        q = col - self.size
        r = row - self.size
        return (q, -r)
