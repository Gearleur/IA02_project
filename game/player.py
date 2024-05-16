import ast

class Player:
    def __init__(self, color):
        self.color = color

    def find_valid_move(self, game):
        for q in range(-game.board.size, game.board.size + 1):
            for r in range(-game.board.size, game.board.size + 1):
                s = -q - r
                if game.is_valid_move(q, r, s):
                    return q, r, s
        return None

    def strategy_brain(self, game):
        game.board.display()
        while True:
            print(f"{self.color} player, enter your move (q, r, s): ", end="")
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
    def strategy_brain(self, game):
        # Simple AI strategy: pick the first valid move
        for q in range(-game.board.size, game.board.size + 1):
            for r in range(-game.board.size, game.board.size + 1):
                s = -q - r
                if game.is_valid_move(q, r, s):
                    return q, r, s
        return None