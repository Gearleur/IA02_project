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

    # Initialisation du jeu
    gopher = GopherGame(board_size=6)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Création de l'état initial et des mouvements pour simuler un état de jeu
    state = gopher.get_initial_state()
    #test state to index
    state = gopher.get_next_state(state, (0, 5, -5), 1)
    state = gopher.get_next_state(state, (0, 4, -4), -1)
    state = gopher.get_next_state(state, (0, 3, -3), 1)
    moves = gopher.get_valid_moves(state, -1)
    tableau = np.zeros((11, 11))
    print(state)
    for move in moves:
        x, y = move
        tableau[x][y] = 1
    print(tableau)
    
    gopher.display(state)
    
    
    
    
    # encoded_state = gopher.get_encoded_states(state)
    # print(encoded_state)
    
    # tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

    # # Charger un modèle spécifique (par exemple, le modèle 2)
    # model_index = 2
    # model_path = f'models/model_{model_index}.pt'
    # optimizer_path = f'models/optimizer_{model_index}.pt'

    # # Initialisation du modèle
    # model = ResNet(gopher, num_resBlocks=9, num_hidden=128, device=device)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    
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
    