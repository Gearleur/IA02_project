# gopher_game/game/game_logic.py
from .hex import Hex, Point, hex_add, hex_subtract, hex_neighbor
from .player import Player
import random
from typing import List, Tuple

class GopherGame:
    def __init__(self, board_size=6):
        self.size = board_size - 1
        self.grid = {}  # Utiliser un dictionnaire pour stocker les hexagones et leurs états
        self.current_player = 1  # 1 pour le premier joueur, 2 pour le second joueur
        self.is_first_turn = True  # Pour suivre le premier tour

    def switch_player(self):
        self.current_player = 3 - self.current_player  # Alterne entre 1 et 2

    def is_valid_move(self, q, r, s):
        hex_pos = Hex(q, r, s)
        
        # Vérifier si le mouvement est à une case vide et qu'il est dans les limites du plateau
        if hex_pos in self.grid or abs(q) > self.size or abs(r) > self.size or abs(s) > self.size:
            return False
        
        # Première tour pour le premier joueur
        if self.is_first_turn and self.current_player == 1:
            return True
        
        # Vérifier les connexions ennemies et amicales selon les règles du jeu
        has_enemy_connection = False
        has_friendly_connection = False
        
        for direction in range(6):
            neighbor = hex_neighbor(hex_pos, direction)
            if neighbor in self.grid:
                if self.grid[neighbor] == self.current_player:
                    has_friendly_connection = True
                else:
                    has_enemy_connection = True
        
        return has_enemy_connection and not has_friendly_connection

    def play_turn(self, q, r, s):
        if self.is_valid_move(q, r, s):
            self.place_stone(q, r, s, self.current_player)
            self.switch_player()
            if self.is_first_turn:
                self.is_first_turn = False  # Fin du premier tour
            if self.has_winner():
                self.display()
                print(f"Player {self.current_player} wins!")
                return True
        else:
            print("Invalid move. Try again.")
        return False
    
    def get_all_valid_moves(self):
        valid_moves = set()
        if not self.grid:
            # Premier tour, toutes les cases sont disponibles
            for q in range(-self.size, self.size + 1):
                for r in range(-self.size, self.size + 1):
                    s = -q - r
                    if abs(q) <= self.size and abs(r) <= self.size and abs(s) <= self.size:
                        valid_moves.add((q, r, s))
        else:
            for hex_pos in self.grid:
                for direction in range(6):
                    neighbor = hex_neighbor(hex_pos, direction)
                    if neighbor not in self.grid and abs(neighbor.q) <= self.size and abs(neighbor.r) <= self.size and abs(neighbor.s) <= self.size:
                        has_enemy_connection = False
                        has_friendly_connection = False
                        for neighbor_dir in range(6):
                            adjacent = hex_neighbor(neighbor, neighbor_dir)
                            if adjacent in self.grid:
                                if self.grid[adjacent] == self.current_player:
                                    has_friendly_connection = True
                                else:
                                    has_enemy_connection = True
                        if has_enemy_connection and not has_friendly_connection:
                            valid_moves.add((neighbor.q, neighbor.r, neighbor.s))
        return list(valid_moves)

    def has_winner(self):
        # on fait get_valide_moves pour le joueur actuel
        current_player_moves = self.get_all_valid_moves()
        #si le joueur actuel n'a pas de mouvement valide, l'autre joueur gagne
        if not current_player_moves:
            self.switch_player()
            return True

    def place_stone(self, q, r, s, player_id):
        hex_pos = Hex(q, r, s)
        if hex_pos not in self.grid:
            self.grid[hex_pos] = player_id
            return True
        return False

    def display(self):
        for r in range(-self.size, self.size + 1):
            # Calculer l'indentation pour chaque ligne
            indent = abs(r)
            print(' ' * indent, end='')
            for q in range(-self.size, self.size + 1):
                s = -q - r
                if abs(q) <= self.size and abs(r) <= self.size and abs(s) <= self.size:
                    hex_pos = Hex(q, r, s)
                    if hex_pos in self.grid:
                        # Afficher 'R' pour le joueur 1 et 'B' pour le joueur 2
                        print('R' if self.grid[hex_pos] == 1 else 'B', end=' ')
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