import numpy as np

toutes_positions=[(3, 3), (0, 3), (1, 3), (2, 3),(2, 1), (3, 0), (3, 1), (3, 2), (0, 2), (1, 2), (2, 2), (1, 1), (2, 0),(-3, -3), (0, -3), (-1, -3), (-2, -3),(-2, -1), (-3, 0), (-3, -1), (-3, -2), (0, -2), (-1, -2),
                          (-2, -2), (-1, -1), (-2, 0), (-1,2),(-2,1),(-1,1),(-1,0),(-2,0),(-3,0),(2,-1),(1,-2),(1,-1),(0,-1),(0,-2),(0,-3)]

directionB=[(-1,0),(-1,-1),(0,-1)]
directionR=[(1,0),(1,1),(0,1)]
testa=(0,1)
testb=(1,0)
print(testa+testb)
class DodoGame:
    def __init__(self, board_size=4):
        self.size = board_size - 1
        self.current_player = 1
        self.action_size = (2 * self.size + 1) ** 2
    def __repr__(self):
        return "DodoGame"

    def get_initial_state(self):

        Position_bleu = [(3, 3), (0, 3), (1, 3), (2, 3),(2, 1), (3, 0), (3, 1), (3, 2), (0, 2), (1, 2), (2, 2), (1, 1), (2, 0)]
        Position_rouge = [(-3, -3), (0, -3), (-1, -3), (-2, -3),(-2, -1), (-3, 0), (-3, -1), (-3, -2), (0, -2), (-1, -2),
                          (-2, -2), (-1, -1), (-2, 0)]

        grid= np.zeros((2 * self.size +1, 2 * self.size +1), dtype=np.int8)
        for x, y in Position_bleu:
            grid[abs(x-self.size),abs(y+self.size)] = -1

        # Set 'r' (rouge) at position_rouge
        for x, y in Position_rouge:
            grid[abs(self.size - x), self.size + y] = 1
        print(grid)
        return grid

    def get_current_player(self, state):
        return 1 if np.sum(state) % 2 == 0 else -1

    def get_next_state(self, state, action, player):
        q, r, s = action
        row = r + self.size
        col = q + self.size
        state[row, col] = player
        return state

    def get_next_state_idx(self, state, action, player):
        row, col = action
        state[row, col] = player
        return state

    def get_next_state_encoded(self, state, action, player):
        rows, cols = 2 * self.size + 1, 2 * self.size + 1
        row, col = np.unravel_index(action, (rows, cols))
        state[row, col] = player
        return state

#raisonnement en matrice et non hexa : mouvement valide si l'action est sur une case vide a cote d'un pion de son equipe
    def is_valid_move(self, grid, action, pion, player=None):
        if player is None:
            player = self.get_current_player(grid)

        # Vérifiez si le mouvement est à l'intérieur des limites de l'hexagone
        if abs(action[0]) > 2*self.size+1 or abs(action[1]) > 2*self.size+1:
            return False

        if grid[action[0]][action[1]] != 0:
            return False

        if grid[pion[0]][pion[1]] != player:
            return False

        has_friendly_connection = False
        for d in directionR:
            if action[0]==d[0]+pion[0] and action[1]==d[1]+pion[1]:
                has_friendly_connection = True
        return has_friendly_connection

#raisonnement en matrice : ici pour le joueur concerne
    def get_valid_moves(self, grid, player=None):
        valid_moves = []
        size = self.size
        if player is None:
            current_player = self.get_current_player(grid)
        else:
            current_player = player
        for pos in toutes_positions:
            for act in toutes_positions:
                if self.is_valid_move(grid,act,pos,player):
                    valid_moves.append([pos,act])
        return valid_moves

    def get_valid_moves_encoded(self, state, player=None):
        board_size = self.size
        valid_moves_encoded = np.zeros((2 * board_size + 1, 2 * board_size + 1), dtype=np.float32)
        valid_moves = self.get_valid_moves(state, player)

        for (x, y) in valid_moves:
            valid_moves_encoded[x, y] = 1

        return valid_moves_encoded.flatten()

#ok!!!!!
    def check_win(self, state, action, player):
        return len(self.get_valid_moves(state, player=player)) == 0

    def get_value_and_terminated(self, state, action, player=None):
        if player is None:
            player = self.get_current_player(state)

        if self.check_win(state, action, player):
            return player, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        return player * state

    def get_encoded_state(self, state):
        board_size = self.size
        current_player = self.get_current_player(state)
        opponent = -current_player

        # Crée les couches pour l'état encodé
        layer_player = (state == current_player).astype(np.float32)
        layer_opponent = (state == opponent).astype(np.float32)
        layer_valid_moves = np.zeros((2 * board_size + 1, 2 * board_size + 1), dtype=np.float32)

        valid_moves = self.get_valid_moves(state)
        for (x, y) in valid_moves:
            layer_valid_moves[x, y] = 1

        # Empile les couches
        encoded_state = np.stack((layer_opponent, layer_valid_moves, layer_player), axis=0)

        return encoded_state

    def get_encoded_states(self, states):
        encoded_states = [self.get_encoded_state(state) for state in states]
        return np.array(encoded_states)
'''
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

'''
dodo=DodoGame()
dodo.get_initial_state()
