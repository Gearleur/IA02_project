import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from game.game_logic import GopherGame, DodoGame
from game.player import Player, AIPlayer, RandomPlayer
from game.hex import oddr_to_axial, axial_to_oddr
from game.mcts_alpha import ResNet, ResBlock, NodeAlpha, MCTSAlpha

    
def main_alpha_gopher():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialisation du jeu
    gopher = GopherGame(board_size=6)

    # # Charger un modèle spécifique (par exemple, le modèle 2)
    # model_index = 7
    # model_path = f'models/model_{model_index}.pt'
    # optimizer_path = f'models/optimizer_{model_index}.pt'

    # # Initialisation du modèle
    # model = ResNet(gopher, num_resBlocks=4, num_hidden=64, device=device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()

    # Création de l'état initial et des mouvements pour simuler un état de jeu
    state = gopher.get_initial_state()
    gopher.display(state)
    
    
    #test state to index
    state = gopher.get_next_state(state, (5, -5, 0), 1)
    state = gopher.get_next_state(state, (4, -4, 0), 2)
    state = gopher.get_next_state(state, (3, -3, 0), 1)
    state = gopher.get_next_state(state, (2, -2, 0), 2)
    state = gopher.get_next_state(state, (1, -2, 1), 1)
    state = gopher.get_next_state(state, (0, -1, 1), 2)
    state = gopher.get_next_state(state, (0, 0, 0), 1)
    gopher.display(state)
    

    # Encoder l'état
    encoded_state = gopher.get_encoded_state(state)
    print(encoded_state)
    
    state = gopher.get_next_state(state, (-1, 1, 0), 2)
    encoded_state = gopher.get_encoded_state(state)
    gopher.display(state)
    print(encoded_state)
    # print(encoded_state)
    # tensor_state = torch.tensor(encoded_state).unsqueeze(0).float()
    # model = ResNet(gopher, num_resBlocks=4, num_hidden=64)

    # # Prédiction avec le modèle
    # policy, value = model(tensor_state)
    # value = value.item()
    # policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    # # Afficher les résultats
    # print(f'Value: {value}')
    # print(f'Policy: {policy}')
    # print(f'valide moves: {gopher.get_valid_moves_encoded(state)}')

    # plt.bar(range(gopher.action_size), policy)
    # plt.show()

if __name__ == "__main__":
    main_alpha_gopher()
    