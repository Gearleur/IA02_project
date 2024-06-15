import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from game import *

# Initialiser le jeu
gopher = GopherGame(board_size=6)

args = {
    'num_searches': 3000,  # Nombre de recherches par itération
    'C': 2  # Constante de contrôle pour l'exploration
}

model = ResNet(gopher, num_resBlocks=4, num_hidden=64)
model.eval()


# Initialiser MCTS
mcts = MCTSAlpha(game=gopher, args=args, model=model)


# Obtenir l'état initial
state = gopher.get_initial_state()
player = 1

while True:
    gopher.display(state)
    if player == 1:
        mcts_probs = mcts.search(state)
        action = np.argmax(mcts_probs)
        state = gopher.get_next_state_encoded(state, action, player)
    else:
        q, r, s = random.choice(gopher.get_valid_moves(state))
        print(q, r, s)
        action = (q, r, s)
        state = gopher.get_next_state(state, action, player)
    #current player,
    
    value, is_terminal = gopher.get_value_and_terminated(state, action)
    
    if is_terminal:
        gopher.display(state)
        print(f"Game over! Player {3-state.current_player}wins!")
        break
    
    player = gopher.get_opponent(player)
