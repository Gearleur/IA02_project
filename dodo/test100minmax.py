import numpy as np
from collections import namedtuple
import random

# Directions for blue and red players
DIRECTIONS = {
    'B': [(1, 0), (1, -1), (0, -1)],
    'R': [(-1, 0), (-1, 1), (0, 1)]
}

Hex = namedtuple("Hex", ["q", "r", "s"])

def hex_to_idx(hex, board_size):
    return hex.r + board_size, hex.q + board_size

class DodoGame:
    def __init__(self, board_size=4):
        self.size = board_size - 1
        self.current_player = 1

    def __repr__(self):
        return "DodoGame"

    def get_initial_state(self):
        # Predefined positions for blue and red players
        positions_bleu = [(0, 4), (0, 3), (0, 5), (0, 6), (1, 3), (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]
        positions_rouge = [(3, 0), (3, 1), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2), (5, 3), (6, 0), (6, 1), (6, 2), (6, 3)]

        grid = np.zeros((2 * self.size + 1, 2 * self.size + 1), dtype=np.int8)
        for x, y in positions_bleu:
            grid[x, y] = -1
        for x, y in positions_rouge:
            grid[x, y] = 1
        return grid

    def get_current_player(self, state):
        return 1 if np.sum(state) % 2 == 0 else -1

    def get_next_state(self, state, action, pion, player):
        new_state = state.copy()
        new_state[action[0], action[1]] = player
        new_state[pion[0], pion[1]] = 0
        return new_state

    def is_valid_move(self, grid, action, pion, player=None):
        if player is None:
            player = self.get_current_player(grid)

        if not (0 <= action[0] < 2 * self.size + 1 and 0 <= action[1] < 2 * self.size + 1):
            return False

        if grid[action[0]][action[1]] != 0 or grid[pion[0]][pion[1]] != player:
            return False

        directions = DIRECTIONS['R'] if player == 1 else DIRECTIONS['B']
        return any(action[0] == d[0] + pion[0] and action[1] == d[1] + pion[1] for d in directions)

    def get_valid_moves(self, grid, player=None):
        if player is None:
            player = self.get_current_player(grid)

        valid_moves = []
        for pos_x in range(2 * self.size + 1):
            for pos_y in range(2 * self.size + 1):
                if grid[pos_x, pos_y] == player:
                    pion = [pos_x, pos_y]
                    for d in (DIRECTIONS['R'] if player == 1 else DIRECTIONS['B']):
                        action = [pos_x + d[0], pos_y + d[1]]
                        if self.is_valid_move(grid, action, pion, player):
                            valid_moves.append([pion, action])
        return valid_moves

    def check_win(self, state, player):
        return len(self.get_valid_moves(state, player=player)) == 0

    def get_value_and_terminated(self, state, player=None):
        if player is None:
            player = self.get_current_player(state)

        if self.check_win(state, player):
            return player, True
        return 0, False

    def display(self, state):
        board_size = self.size
        for r in range(-board_size, board_size + 1):
            indent = abs(r)
            print(' ' * indent, end='')
            for q in range(-board_size, board_size + 1):
                s = -q - r
                if abs(q) <= board_size and abs(r) <= board_size and abs(s) <= board_size:
                    x, y = hex_to_idx(Hex(q, r, s), board_size)
                    if state[x][y] == 1:
                        print('R', end=' ')
                    elif state[x][y] == -1:
                        print('B', end=' ')
                    else:
                        print('.', end=' ')
            print()

def evaluate(state, player):
    return np.sum(state == -player)

def minimax(game, state, depth, alpha, beta, maximizing_player):
    current_player = game.get_current_player(state)
    value, terminated = game.get_value_and_terminated(state, current_player)
    if depth == 0 or terminated:
        return evaluate(state, current_player), None

    valid_moves = game.get_valid_moves(state, current_player)
    best_move = None

    if maximizing_player:
        max_eval = -float('inf')
        for move in valid_moves:
            new_state = game.get_next_state(state, move[1], move[0], current_player)
            eval, _ = minimax(game, new_state, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in valid_moves:
            new_state = game.get_next_state(state, move[1], move[0], current_player)
            eval, _ = minimax(game, new_state, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

def random_move(game, state, player):
    valid_moves = game.get_valid_moves(state, player)
    return random.choice(valid_moves) if valid_moves else None

def play_game(display_game=False):
    game = DodoGame()
    state = game.get_initial_state()
    current_player = 1

    while True:
        if display_game:
            game.display(state)
        value, terminated = game.get_value_and_terminated(state, current_player)
        if terminated:
            return value

        if current_player == 1:
            _, move = minimax(game, state, 10, -float('inf'), float('inf'), True)
        else:
            move = random_move(game, state, current_player)

        if move is not None:
            state = game.get_next_state(state, move[1], move[0], current_player)
        current_player *= -1

def simulate_games(num_games=10):
    red_wins = 0
    blue_wins = 0

    for _ in range(num_games):
        result = play_game()
        if result == 1:
            red_wins += 1
        elif result == -1:
            blue_wins += 1

    print(f"Red (Minimax) won {red_wins} times.")
    print(f"Blue (Random) won {blue_wins} times.")

if __name__ == "__main__":
    simulate_games(1)
