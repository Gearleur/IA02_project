import ast
from game.hex import Hex, Point, hex_neighbor, hex_add, hex_subtract

class Player:

    def find_valid_move(self, game)->tuple:
        for q in range(-game.board.size, game.board.size + 1):
            for r in range(-game.board.size, game.board.size + 1):
                s = -q - r
                if game.is_valid_move(q, r, s):
                    return q, r, s
        return None

    def strategy(self, game)->tuple:
        game.board.display()
        while True:
            print(f"{game.get_current_player().color} player, enter your move (q, r, s): ", end="")
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
                
                
class AIPlayer(Player):
    
    def strategy(self, game):
        # Appel à l'algorithme Minimax pour déterminer le meilleur coup
        best_move = self.minimax(game, depth=3, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
        return best_move[1]

    def minimax(self, game, depth, alpha, beta, maximizing_player):
        if depth == 0 or game.has_winner():
            return self.evaluate_board(game), None
        
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for q in range(-game.board.size, game.board.size + 1):
                for r in range(-game.board.size, game.board.size + 1):
                    s = -q - r
                    if game.is_valid_move(q, r, s):
                        game.board.place_stone(q, r, s, game.get_current_player().color)
                        eval = self.minimax(game, depth-1, alpha, beta, False)[0]
                        game.board.grid.pop(Hex(q, r, s))  # Undo move
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
            for q in range(-game.board.size, game.board.size + 1):
                for r in range(-game.board.size, game.board.size + 1):
                    s = -q - r
                    if game.is_valid_move(q, r, s):
                        game.board.place_stone(q, r, s, game.get_current_player().color)
                        eval = self.minimax(game, depth-1, alpha, beta, True)[0]
                        game.board.grid.pop(Hex(q, r, s))  # Undo move
                        if eval < min_eval:
                            min_eval = eval
                            best_move = (q, r, s)
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval, best_move

    def evaluate_board(self, game):
        # Simple evaluation function counting valid moves
        current_player_color = game.get_current_player().color
        opponent_color = 'B' if current_player_color == 'R' else 'R'
        current_player_moves = sum(
            1 for q in range(-game.board.size, game.board.size + 1)
            for r in range(-game.board.size, game.board.size + 1)
            for s in [-q - r]
            if game.is_valid_move(q, r, s)
        )
        opponent_moves = sum(
            1 for q in range(-game.board.size, game.board.size + 1)
            for r in range(-game.board.size, game.board.size + 1)
            for s in [-q - r]
            if game.is_valid_move(q, r, s)
        )
        return current_player_moves - opponent_moves