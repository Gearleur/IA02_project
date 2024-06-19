import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
model.load_state_dict(torch.load("model_0_GopherGame.pt", map_location=device))
model.eval()


# Initialiser MCTS
mcts = MCTSAlpha(game=gopher, args=args, model=model)

def simulate_game(mcts, gopher):
    state = gopher.get_initial_state()
    player = 1
    
    while True:
        if player == 1:
            mcts_probs = mcts.search(state)
            action = np.argmax(mcts_probs)
            state = gopher.get_next_state_encoded(state, action, player)
        else:
            action = random.choice(gopher.get_valid_moves(state))
            state = gopher.get_next_state_idx(state, action, player)
        
        value, is_terminal = gopher.get_value_and_terminated(state, player)
        
        if is_terminal:
            return player, value
        
        player = gopher.get_opponent(player)

def run_simulations(num_games, mcts, gopher):
    mcts_wins = 0
    
    for _ in range(num_games):
        player, value = simulate_game(mcts, gopher)
        if player == 1 and value == 1:  # Vérifiez si MCTS (Player 1) a gagné
            mcts_wins += 1
    
    return mcts_wins

# Exécuter les simulations
num_games = 25
mcts_wins = run_simulations(num_games, mcts, gopher)

print(f"MCTS a gagné {mcts_wins} parties sur {num_games}.")
