import numpy as np
from collections import namedtuple

toutes_positions = [(3, 3), (0, 3), (1, 3), (2, 3), (2, 1), (3, 0), (3, 1), (3, 2), (0, 2), (1, 2), (2, 2), (1, 1), (2, 0),
                    (-3, -3), (0, -3), (-1, -3), (-2, -3), (-2, -1), (-3, 0), (-3, -1), (-3, -2), (0, -2), (-1, -2),
                    (-2, -2), (-1, -1), (-2, 0), (-1, 2), (-2, 1), (-1, 1), (-1, 0), (-2, 0), (-3, 0), (2, -1), (1, -2),
                    (1, -1), (0, -1), (0, -2), (0, -3)]

directionB = [(1, 0), (1, -1), (0, -1)]
directionR = [(-1, 0), (-1, 1), (0, 1)]

Point = namedtuple("Point", ["x", "y"])
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
        Position_bleu = [(0, 4), (0, 3), (0, 5), (0, 6), (1, 3), (1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]
        Position_rouge = [(3, 0), (3, 1), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (5, 2), (5, 3), (6, 0),
                          (6, 1), (6, 2), (6, 3)]

        grid = np.zeros((2 * self.size + 1, 2 * self.size + 1), dtype=np.int8)
        for x, y in Position_bleu:
            grid[x, y] = -1

        for x, y in Position_rouge:
            grid[x, y] = 1
        return grid

    def get_current_player(self, state):
        return 1 if np.sum(state) % 2 == 0 else -1

    def get_next_state(self, state, action, pion, player):
        state[action[0], action[1]] = player
        state[pion[0], pion[1]] = 0
        return state

    def is_valid_move(self, grid, action, pion, player=None):
        caseInterdite = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (4, 6), (2, 0), (5, 6), (6, 6), (6, 5), (6, 4), (5, 5)]

        if player is None:
            player = self.get_current_player(grid)

        if not (0 <= action[0] < 2 * self.size + 1 and 0 <= action[1] < 2 * self.size + 1):
            return False

        if grid[action[0]][action[1]] != 0 or grid[pion[0]][pion[1]] != player:
            return False

        if tuple(action) in caseInterdite:
            return False

        directions = directionR if player == 1 else directionB
        return any(action[0] == d[0] + pion[0] and action[1] == d[1] + pion[1] for d in directions)

    def get_valid_moves(self, grid, player=None):
        valid_moves = []
        if player is None:
            current_player = self.get_current_player(grid)
        else:
            current_player = player

        for pos_x in range(2 * self.size + 1):
            for pos_y in range(2 * self.size + 1):
                pos = [pos_x, pos_y]
                for act_x in range(2 * self.size + 1):
                    for act_y in range(2 * self.size + 1):
                        act = [act_x, act_y]
                        if self.is_valid_move(grid, act, pos, player):
                            valid_moves.append([pos, act])
        return valid_moves

    def check_win(self, state, action, player):
        return len(self.get_valid_moves(state, player=player)) == 0

    def get_value_and_terminated(self, state, action, player=None):
        if player is None:
            player = self.get_current_player(state)

        if self.check_win(state, action, player):
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
                    if state[x][y] == 1:  # Remarquez que x et y sont inversés pour accéder correctement à state
                        print('R', end=' ')
                    elif state[x][y] == -1:
                        print('B', end=' ')
                    else:
                        print('.', end=' ')
            print()


dodo = DodoGame()
grid = dodo.get_initial_state()
print(dodo.get_valid_moves(grid, 1))

print(dodo.get_next_state(grid, [2, 1], [3, 0], 1))
print(dodo.get_valid_moves(grid, 1))



