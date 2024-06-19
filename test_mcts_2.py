import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from gopher import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialiser le jeu
gopher = GopherGame(board_size=6)

args = {
    'C': 2,
    'num_searches': 600,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}

model = ResNet(gopher, num_resBlocks=9, num_hidden=128, device=device)
model.load_state_dict(torch.load("model_7_GopherGame.pt", map_location=device))
model.eval()


# Initialiser MCTS
mcts = MCTSAlpha(game=gopher, args=args, model=model)

# Obtenir l'état initial
state = gopher.get_initial_state()
player = 1

while True:
    gopher.display(state)
    if player == 1:
        action= random.choice(gopher.get_valid_moves(state))
        state = gopher.get_next_state_idx(state, action, player)
    else:
        neutral_state = gopher.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        state = gopher.get_next_state_encoded(state, action, player)
    #current player,
    
    value, is_terminal = gopher.get_value_and_terminated(state, player)
    
    if is_terminal:
        gopher.display(state)
        print(f"Game over! Player {player}wins!")
        break
    
    player = gopher.get_opponent(player)
