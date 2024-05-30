import ast
import random
from game.hex import Hex, Point, hex_neighbor, hex_add, hex_subtract


class Player:
    def find_valid_move(self, game) -> tuple:
        for q in range(-game.size, game.size + 1):
            for r in range(-game.size, game.size + 1):
                s = -q - r
                if game.is_valid_move(q, r, s):
                    return q, r, s
        return None

    def strategy(self, game) -> tuple:
        """
        La fonction `strategy` affiche le plateau de jeu puis entre dans une boucle pour demander au joueur
        actuel de saisir son coup.

        :param game: L'instance du jeu qui contient le plateau et la logique du jeu.
        """
        game.display()
        while True:
            print(f"Player {game.current_player}, enter your move (q, r, s): ", end="")
            s = input()
            try:
                action = ast.literal_eval(s)
                if isinstance(action, tuple) and len(action) == 3:
                    q, r, s = action
                    if q + r + s == 0:
                        if game.is_valid_move(q, r, s):
                            return q, r, s
                        else:
                            print("Invalid move according to the game rules.")
                            suggestion = self.find_valid_move(game)
                            if suggestion:
                                print(f"Suggested valid move: {suggestion}")
                            else:
                                print("No valid moves available.")
                    else:
                        print("The coordinates must satisfy q + r + s = 0.")
                else:
                    print("Input must be a tuple of three indices (q, r, s).")
            except (SyntaxError, ValueError):
                print("Invalid input. Please enter a valid tuple, e.g., (0, 1, -1)")


                
class AIPlayer:
    def __init__(self, game, depth=3):
        self.game = game
        self.depth = depth
    
    def strategy(self, game):
        # Appel à l'algorithme Minimax pour déterminer le meilleur coup
        best_move = self.minimax(game, depth=self.depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
        if best_move[1] is not None:
            return best_move[1]
        else:
            raise ValueError("No valid moves available")

    def minimax(self, game, depth, alpha, beta, maximizing_player, indent=0):
        if depth == 0 or game.has_winner():
            return self.evaluate_board(game), None
        
        valid_moves = self.get_valid_moves(game)
        if not valid_moves:
            return self.evaluate_board(game), None


        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for move in valid_moves:
                q, r, s = move
                game.place_stone(q, r, s, game.current_player)
                game.switch_player()  # Changer le joueur

                eval = self.minimax(game, depth-1, alpha, beta, False, indent + 2)[0]
                game.grid.pop(Hex(q, r, s))  # Annuler le coup
                game.switch_player()  # Revenir au joueur précédent

                if eval > max_eval:
                    max_eval = eval
                    best_move = (q, r, s)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in valid_moves:
                q, r, s = move
                game.place_stone(q, r, s, game.current_player)
                game.switch_player()  # Changer le joueur

                eval = self.minimax(game, depth-1, alpha, beta, True, indent + 2)[0]
                game.grid.pop(Hex(q, r, s))  # Annuler le coup
                game.switch_player()  # Revenir au joueur précédent

                if eval < min_eval:
                    min_eval = eval
                    best_move = (q, r, s)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def get_valid_moves(self, game):
        valid_moves = set()
        if not game.grid:
            # Premier tour, toutes les cases sont disponibles
            for q in range(-game.size, game.size + 1):
                for r in range(-game.size, game.size + 1):
                    s = -q - r
                    if abs(q) <= game.size and abs(r) <= game.size and abs(s) <= game.size:
                        valid_moves.add((q, r, s))
        else:
            for hex_pos in game.grid:
                for direction in range(6):
                    neighbor = hex_neighbor(hex_pos, direction)
                    if neighbor not in game.grid and abs(neighbor.q) <= game.size and abs(neighbor.r) <= game.size and abs(neighbor.s) <= game.size:
                        has_enemy_connection = False
                        has_friendly_connection = False
                        for neighbor_dir in range(6):
                            adjacent = hex_neighbor(neighbor, neighbor_dir)
                            if adjacent in game.grid:
                                if game.grid[adjacent] == game.current_player:
                                    has_friendly_connection = True
                                else:
                                    has_enemy_connection = True
                        if has_enemy_connection and not has_friendly_connection:
                            valid_moves.add((neighbor.q, neighbor.r, neighbor.s))
        return list(valid_moves)


    def evaluate_board(self, game):
        def is_opposite(hex1, hex2):
            return (hex1.q == -hex2.q) and (hex1.r == -hex2.r) and (hex1.s == -hex2.s)

        current_player_id = game.current_player
        opponent_id = 3 - current_player_id

        current_player_moves = self.get_valid_moves(game)
        game.switch_player()
        opponent_moves = self.get_valid_moves(game)
        game.switch_player()

        # Penalty for playing in an opposite cell
        penalty = 10

        def score_moves(moves, player_id):
            score = 0
            for move in moves:
                for hex_pos in game.grid:
                    if game.grid[hex_pos] == player_id and is_opposite(hex_pos, Hex(*move)):
                        score -= penalty
            return score

        current_player_score = score_moves(current_player_moves, current_player_id)
        opponent_score = score_moves(opponent_moves, opponent_id)

        return current_player_score - opponent_score

class RandomPlayer:
    def find_valid_move(self, game) -> tuple:
        valid_moves = self.get_all_valid_moves(game)
        if valid_moves:
            return random.choice(valid_moves)
        return None

    def get_all_valid_moves(self, game):
        valid_moves = set()
        if not game.grid:
            # Premier tour, toutes les cases sont disponibles
            for q in range(-game.size, game.size + 1):
                for r in range(-game.size, game.size + 1):
                    s = -q - r
                    if abs(q) <= game.size and abs(r) <= game.size and abs(s) <= game.size:
                        valid_moves.add((q, r, s))
        else:
            for hex_pos in game.grid:
                for direction in range(6):
                    neighbor = hex_neighbor(hex_pos, direction)
                    if neighbor not in game.grid and abs(neighbor.q) <= game.size and abs(neighbor.r) <= game.size and abs(neighbor.s) <= game.size:
                        has_enemy_connection = False
                        has_friendly_connection = False
                        for neighbor_dir in range(6):
                            adjacent = hex_neighbor(neighbor, neighbor_dir)
                            if adjacent in game.grid:
                                if game.grid[adjacent] == game.current_player:
                                    has_friendly_connection = True
                                else:
                                    has_enemy_connection = True
                        if has_enemy_connection and not has_friendly_connection:
                            valid_moves.add((neighbor.q, neighbor.r, neighbor.s))
        return list(valid_moves)

    def strategy(self, game) -> tuple:
        """
        La fonction `strategy` choisit un mouvement valide aléatoirement.
        
        :param game: L'instance du jeu qui contient le plateau et la logique du jeu.
        """
        return self.find_valid_move(game)
    
    
