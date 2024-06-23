import argparse
import random
import time
import numpy as np
import torch
from typing import Dict, Any, Tuple
from typing import List, Tuple
from .hex import Hex, hex_neighbor

Cell = Tuple[int, int]


class GopherGame2:
    def __init__(self, board_size=6):
        self.size = board_size - 1
        self.board_size = 2 * self.size + 1
        self.memo = {}

    def __repr__(self):
        return "GopherGame"

    def get_initial_state(self):
        return {}

    def get_current_player(self, state):
        return 1 if len(state) % 2 == 0 else -1

    def get_next_state(self, state, action, player):
        state[action] = player
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
        if action in state:
            return False

        enemy_connections = 0
        has_friendly_connection = False

        for direction in range(6):
            neighbor = hex_neighbor(action, direction)
            if max(abs(neighbor.q), abs(neighbor.r), abs(neighbor.s)) <= self.size:
                if neighbor in state:
                    if state[neighbor] == player:
                        has_friendly_connection = True
                    elif state[neighbor] != 0:
                        enemy_connections += 1
                        if enemy_connections > 1:
                            return False

        return not has_friendly_connection and enemy_connections == 1

    def get_valid_moves(self, state, player=None):
        valid_moves = []
        size = self.size
        if not state:
            for q in range(-size, size + 1):
                for r in range(-size, size + 1):
                    s = -q - r
                    if abs(q) <= size and abs(r) <= size and abs(s) <= size:
                        valid_moves.append(Hex(q, r, s))
        else:
            for q in range(-size, size + 1):
                for r in range(-size, size + 1):
                    s = -q - r
                    if abs(q) <= size and abs(r) <= size and abs(s) <= size:
                        action = Hex(q, r, s)
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

    def is_terminal_state(self, state, player):
        return len(self.get_valid_moves(state, player)) == 0

    def evaluate_state(self, state, player):
        score = 0
        for pos, val in state.items():
            if val == player:
                score += 1
            elif val == -player:
                score -= 1
        return score

    def evaluate_state_terminal(self, state, player):
        return -1000

    def serveur_state_to_gopher(
        self, server_state: List[Tuple[Tuple[int, int], int]]
    ) -> List[List[int]]:
        state = {}
        for cell, value in server_state:
            q = cell[0]
            r = cell[1]
            s = -q + r
            if value == 1:
                state[Hex(q, -r, s)] = 1
            elif value == 2:
                state[Hex(q, -r, s)] = -1
        return state

    def action_to_server(self, action):
        return (action.q, -action.r)
