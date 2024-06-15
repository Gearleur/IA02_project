# gopher_game/game/game_logic.py
from .hex import Hex, Point, hex_add, hex_subtract, hex_neighbor
from .player import Player
import random
import numpy as np
from typing import List, Tuple

class GopherGame:
    def __init__(self, board_size=6):
        self.size = board_size - 1
        self.current_player = 1
        self.action_size = (2 * self.size + 1) ** 2
    
    def __repr__(self):
        return "GopherGame"
        
    def get_initial_state(self):
        return {}
    
    def get_current_player(self, state):
        # Compter le nombre de pions de chaque joueur
        count_player_1 = sum(1 for player in state.values() if player == 1)
        count_player_2 = sum(1 for player in state.values() if player == 2)
        
        # Si c'est au tour du joueur 1 de jouer
        if count_player_1 <= count_player_2:
            return 1
        else:
            return 2
    
    def get_next_state(self, state, action, player):
        q, r, s = action
        if self.is_valid_move(state, q, r, s):
            state[Hex(q, r, s)] = player
        return state
    
    def get_next_state_encoded(self, state, action, player):
        rows, cols = 2 * self.size + 1, 2 * self.size + 1
        row, col = np.unravel_index(action, (rows, cols))
        r = row - (self.size)
        q = col - (self.size)
        s = -q - r
        return self.get_next_state(state, (q, r, s), player)
    
    def get_valid_moves(self, state):
        valid_moves = set()
        if not state:
            for q in range(-self.size, self.size + 1):
                for r in range(-self.size, self.size + 1):
                    s = -q - r
                    if abs(q) <= self.size and abs(r) <= self.size and abs(s) <= self.size:
                        valid_moves.add((q, r, s))
        else:
            for hex_pos in state:
                for direction in range(6):
                    neighbor = hex_neighbor(hex_pos, direction)
                    if neighbor not in state and abs(neighbor.q) <= self.size and abs(neighbor.r) <= self.size and abs(neighbor.s) <= self.size:
                        has_enemy_connection = False
                        has_friendly_connection = False
                        for neighbor_dir in range(6):
                            adjacent = hex_neighbor(neighbor, neighbor_dir)
                            if adjacent in state:
                                if state[adjacent] == self.get_current_player(state):
                                    has_friendly_connection = True
                                else:
                                    has_enemy_connection = True
                        if has_enemy_connection and not has_friendly_connection:
                            valid_moves.add((neighbor.q, neighbor.r, neighbor.s))
        return list(valid_moves)
    
    def get_valid_moves_encoded(self, state):
        board_size = self.size
        valid_moves_encoded = np.zeros((2 * board_size + 1,2 * board_size + 1), dtype=np.float32)
        valid_moves = self.get_valid_moves(state)
        for (q, r, s) in valid_moves:
            if abs(q) > board_size or abs(r) > board_size or abs(s) > board_size:
                continue
            q_idx = q + board_size
            r_idx = r + board_size
            if 0 <= q_idx < 2 * board_size + 1 and 0 <= r_idx < 2 * board_size + 1:
                valid_moves_encoded[r_idx, q_idx] = 1
                
        return valid_moves_encoded.flatten()
    
    def check_win(self, state, action):
        return len(self.get_valid_moves(state)) == 0
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        return 0, False
    
    def get_opponent(self, player):
        return 3 - player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        new_state = state.copy()
        if player == 2:
            new_state = {pos: (3 - p) for pos, p in state.items()}
        return new_state
    
    def get_encoded_state(self, state):
        board_size = self.size
        encoded_state = np.zeros((3, 2 * board_size + 1, 2 * board_size + 1), dtype=np.float32)

        opponent = 3 - self.get_current_player(state)
        for (q, r, s), player in state.items():
            if abs(q) > board_size or abs(r) > board_size or abs(s) > board_size:
                continue
            q_idx = q + board_size
            r_idx = r + board_size
            if 0 <= q_idx < 2 * board_size + 1 and 0 <= r_idx < 2 * board_size + 1:
                if player == opponent:
                    encoded_state[0, r_idx, q_idx] = 1
                elif player == self.get_current_player(state):
                    encoded_state[2, r_idx, q_idx] = 1

        valid_moves = self.get_valid_moves(state)
        for (q, r, s) in valid_moves:
            if abs(q) > board_size or abs(r) > board_size or abs(s) > board_size:
                continue
            q_idx = q + board_size
            r_idx = r + board_size
            if 0 <= q_idx < 2 * board_size + 1 and 0 <= r_idx < 2 * board_size + 1:
                encoded_state[1, r_idx, q_idx] = 1

        return encoded_state
    
    def clone(self):
        new_game = GopherGame(self.size + 1)
        new_game.grid = self.grid.copy()
        new_game.current_player = self.current_player
        new_game.is_first_turn = self.is_first_turn
        return new_game
    
    def is_valid_move(self, state, q, r, s):
        hex_pos = Hex(q, r, s)
        if hex_pos in state or abs(q) > self.size or abs(r) > self.size or abs(s) > self.size:
            return False
        if state == {}:
            return True
        has_enemy_connection = False
        has_friendly_connection = False
        for direction in range(6):
            neighbor = hex_neighbor(hex_pos, direction)
            if neighbor in state:
                if state[neighbor] == self.get_current_player(state):
                    has_friendly_connection = True
                else:
                    has_enemy_connection = True
        return has_enemy_connection and not has_friendly_connection

    def display(self, state):
        for r in range(-self.size, self.size + 1):
            indent = abs(r)
            print(' ' * indent, end='')
            for q in range(-self.size, self.size + 1):
                s = -q - r
                if abs(q) <= self.size and abs(r) <= self.size and abs(s) <= self.size:
                    hex_pos = Hex(q, r, s)
                    if hex_pos in state:
                        print('R' if state[hex_pos] == 1 else 'B', end=' ')
                    else:
                        print('.', end=' ')
            print()
            
    
    
    
            
class DodoGame:
    def __init__(self, hex_size):
        self.hex_size = hex_size
        self.state = self.get_initial_state()
        self.current_player = 1

    def get_initial_state(self) -> List[Tuple[Hex, int]]:
        state = []
        # Setup the board based on hex_size; assuming symmetrical setup
        for q in range(-self.hex_size + 1, self.hex_size):
            for r in range(max(-self.hex_size + 1, -q - self.hex_size + 1), min(self.hex_size, -q + self.hex_size)):
                if q <= 0:
                    state.append((Hex(q, r, -q-r), 1))  # Red player
                else:
                    state.append((Hex(q, r, -q-r), 2))  # Blue player
        return state

    def get_valid_moves(self, state: List[Tuple[Hex, int]], player: int) -> List[Tuple[Hex, Hex]]:
        moves = []
        directions = [0, 1, 2] if player == 1 else [3, 4, 5]  # Forward directions based on player
        for hex, p in state:
            if p == player:
                for d in directions:
                    neighbor = hex_neighbor(hex, d)
                    if self.is_cell_free(neighbor, state):
                        moves.append((hex, neighbor))
        return moves

    def is_cell_free(self, hex: Hex, state: List[Tuple[Hex, int]]) -> bool:
        return all(hex.q != x.q or hex.r != x.r for x, _ in state)

    def make_move(self, move: Tuple[Hex, Hex]):
        start, end = move
        for i, (hex, p) in enumerate(self.state):
            if hex == start:
                self.state[i] = (end, p)
                break
        self.current_player = 3 - self.current_player  # Switch player

    def check_win(self, state: List[Tuple[Hex, int]], player: int) -> bool:
        return not self.get_valid_moves(state, player)
    
    def display(self):
        # Create a dictionary for quick access to the state
        state_dict = {(hex.q, hex.r): player for hex, player in self.state}
        # Display the board
        print("\n" + " " * (2 * self.hex_size - 1) + "*")
        for r in range(-self.hex_size + 1, self.hex_size):
            offset = abs(r)
            row = []
            for q in range(-self.hex_size + 1, self.hex_size):
                if (q, r) in state_dict:
                    if state_dict[(q, r)] == 1:
                        row.append('R')  # Red player
                    elif state_dict[(q, r)] == 2:
                        row.append('B')  # Blue player
                else:
                    row.append(' ')
            print(" " * offset + ' '.join(row))
        print(" " * (2 * self.hex_size - 1) + "*\n")
