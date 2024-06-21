from .hex import Hex, Point, hex_add, hex_subtract, hex_neighbor, idx_to_hex, hex_to_idx
import random
import numpy as np
from typing import List, Tuple

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

        has_enemy_connection = False
        has_friendly_connection = False

        for direction in range(6):
            neighbor = hex_neighbor(action, direction)
            if (
                abs(neighbor.q) <= self.size
                and abs(neighbor.r) <= self.size
                and abs(neighbor.s) <= self.size
            ):
                if neighbor in state:
                    if state[neighbor] == player:
                        has_friendly_connection = True
                    elif state[neighbor] != 0:
                        has_enemy_connection = True

        return not has_friendly_connection and has_enemy_connection

    
    def get_valid_moves(self, state, player=None):
        valid_moves = []
        size = self.size
        is_empty = not bool(state)  # Vérifie si l'état est vide
        if player is None:
            current_player = self.get_current_player(state)
        else:
            current_player = player

        if is_empty:
            for q in range(-size, size + 1):
                for r in range(-size, size + 1):
                    s = -q - r
                    if abs(q) <= size and abs(r) <= size and abs(s) <= size:
                        valid_moves.append(Hex(q, r, s))
        else:
            for q in range(-size, size + 1):
                for r in range(-size, size + 1):
                    s = -q - r
                    action = Hex(q, r, s)
                    if self.is_valid_move(state, action, current_player):
                        valid_moves.append(action)
        return valid_moves
            
    
    def display(self, state):
        for r in range(-self.size, self.size+ 1):
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
        # Exemple d'une fonction d'évaluation simple
        score = 0
        for pos, val in state.items():
            if val == player:
                score += 1
            elif val == -player:
                score -= 1
        return score
    
    