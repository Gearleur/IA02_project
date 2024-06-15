import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from game.game_logic import GopherGame, DodoGame
from game.player import Player, AIPlayer, RandomPlayer
from game.hex import oddr_to_axial, axial_to_oddr
from game.mcts import Node, MCTS

# Définir les arguments pour MCTS
args = {
    'num_searches': 1000,  # Nombre de recherches par itération
    'C': 1.4  # Constante de contrôle pour l'exploration
}

# Initialiser le jeu
gopher = GopherGame(board_size=6)

# Initialiser MCTS
mcts = MCTS(game=gopher, args=args)

# Obtenir l'état initial
state = gopher.get_initial_state()
player = 1

while True:
    gopher.display(state)
    if player == 1:
        q, r, s = random.choice(gopher.get_valid_moves(state))
        print(q, r, s)
        action = (q, r, s)
        state = gopher.get_next_state(state, action, player)
    else:
        neutral_state = gopher.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        state = gopher.get_next_state_encoded(state, action, player)
    #current player,
    
    value, is_terminal = gopher.get_value_and_terminated(state, action)
    
    if is_terminal:
        gopher.display(state)
        print(f"Game over! Player {3-state.current_player}wins!")
        break
    
    player = gopher.get_opponent(player)
