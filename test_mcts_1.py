import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from gopher import *

# Définir les arguments pour MCTS
args = {
    'C': 2,
    'num_searches': 600,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}
# Initialiser le jeu
gopher = GopherGame(board_size=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model random pour essayer
mock_resnet = MockResNet(gopher, num_resBlocks=5, num_hidden=128, device=device)
# Initialiser MCTS
mcts = MCTSAlpha(game=gopher, args=args, model=mock_resnet)

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
        action= random.choice(gopher.get_valid_moves(state))
        state = gopher.get_next_state_idx(state, action, player)
    #current player,
    
    value, is_terminal = gopher.get_value_and_terminated(state, action)
    
    if is_terminal:
        gopher.display(state)
        print(f"Game over! Player {-player}wins!")
        break
    
    player = -player
