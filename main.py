import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gopher import *

    
def main_alpha_gopher():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialiser le jeu
    gopher = GopherGame(board_size=6)

    args = {
        'C': 2,
        'num_searches': 600,
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }

    model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
    model.load_state_dict(torch.load("model_0_GopherGame.pt", map_location=device))
    model.eval()


    # Initialiser MCTS
    mcts = MCTSAlpha(game=gopher, args=args, model=model)
    state = gopher.get_initial_state()
    state = gopher.get_next_state(state, (-4, 0, 4), 1)
    neutral_state = gopher.change_perspective(state, -1)
    action = np.argmax(mcts.search(neutral_state))
    state = gopher.get_next_state_encoded(state, action, -1)
    gopher.display(state)
    
    
    
    
if __name__ == "__main__":
    main_alpha_gopher()
    