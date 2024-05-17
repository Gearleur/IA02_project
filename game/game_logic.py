# gopher_game/game/game_logic.py

from .board import Board_gopher
from .hex import Hex, Point, hex_add, hex_subtract, hex_neighbor 
from .player import Player

class GopherGame:
    def __init__(self, player1, player2, board_size=6):
        self.board = Board_gopher(size=board_size)
        self.players = [player1, player2]
        self.players[0].color = 'R'  # Assigner Rouge au premier joueur
        self.players[1].color = 'B'  # Assigner Bleu au second joueur
        self.current_player_index = 0  # Index du joueur actuel, 0 pour Rouge, 1 pour Bleu
        self.is_first_turn = True  # Pour suivre le premier tour
        
    def switch_player(self)->None:
        self.current_player_index = 1 - self.current_player_index

    def get_current_player(self)->Player:
        return self.players[self.current_player_index]

    def is_valid_move(self, q, r, s)->bool:
        # Vérifier si le mouvement est à une case vide et qu'il est dans les limites du plateau
        if not self.board.is_valid_move(q, r, s):
            return False
        
        # Première tour pour Rouge
        if self.is_first_turn and self.get_current_player().color == 'R':
            return True
        
        # Vérifier les connexions ennemies et amicales selon les règles du jeu
        has_enemy_connection = False
        has_friendly_connection = False
        
        for direction in range(6):
            neighbor = hex_neighbor(Hex(q, r, s), direction)
            if neighbor in self.board.grid:
                if self.board.grid[neighbor] == self.get_current_player().color:
                    has_friendly_connection = True
                else:
                    has_enemy_connection = True
        
        return has_enemy_connection and not has_friendly_connection

    def play_turn(self)->bool:
        player = self.get_current_player()
        q, r, s = player.strategy(self)
        if self.is_valid_move(q, r, s):
            self.board.place_stone(q, r, s, player.color)
            if self.is_first_turn:
                self.is_first_turn = False  # Fin du premier tour
            self.switch_player()
            if self.has_winner():
                self.board.display()
                print(f"Player {player.color} wins!")
                return True
        else:
            print("Invalid move. Try again.")
        return False

    def has_winner(self)->bool:
        # La condition de victoire est si le joueur ne peut plus placer de pierre
        current_player_color = self.get_current_player().color
        for q in range(-self.board.size, self.board.size + 1):
            for r in range(-self.board.size, self.board.size + 1):
                s = -q - r
                if abs(q) <= self.board.size and abs(r) <= self.board.size and abs(s) <= self.board.size:
                    if self.is_valid_move(q, r, s):
                        return False
        return True