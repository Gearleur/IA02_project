import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gopher import *
from dodo import *


def main_alpha_gopher():
    # Définir les arguments pour MCTS
    args = {"C": 2, "num_searches": 1200, "dirichlet_epsilon": 0.0, "dirichlet_alpha": 0.3}
    # Initialiser le jeu
    gopher = GopherGame(board_size=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model random pour essayer
    model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
    model.load_state_dict(torch.load("model_0_GopherGame.pt", map_location=device))
    # Initialiser MCTS
    mcts = MCTSAlpha(game=gopher, args=args, model=model)

    # Obtenir l'état initial
    state = gopher.get_initial_state()
    player = 1

    while True:
        gopher.display(state)
        if player == 1:
            action = random.choice(gopher.get_valid_moves(state))
            state = gopher.get_next_state_idx(state, action, player)
        else:
            neutral_state = gopher.change_perspective(state, -1)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            state = gopher.get_next_state_encoded(state, action, player)
        # current player,

        value, is_terminal = gopher.get_value_and_terminated(state, player)

        if is_terminal:
            gopher.display(state)
            (print(gopher.get_valid_moves(state, 1)))
            (print(gopher.get_valid_moves(state, -1)))
            print(f"Game over! Player {player} wins!")
            break

        player = -player

def main_rng_gopher(num_games=25):
    # Définir les arguments pour MCTS
    args = {"C": 2, "num_searches": 1200, "dirichlet_epsilon": 0.0, "dirichlet_alpha": 0.3}
    # Initialiser le jeu
    gopher = GopherGame(board_size=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model random pour essayer
    model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
    model.load_state_dict(torch.load("model_0_GopherGame.pt", map_location=device))
    # Initialiser MCTS
    mcts = MCTSAlpha(game=gopher, args=args, model=model)

    mcts_wins = 0
    rng_wins = 0

    for game in range(num_games):
        # Obtenir l'état initial
        state = gopher.get_initial_state()
        # Déterminer l'ordre des joueurs
        mcts_first = game % 2 == 0
        player = 1

        while True:
            if player == 1:
                if mcts_first:
                    mcts_probs = mcts.search(state)
                    action = np.argmax(mcts_probs)
                    state = gopher.get_next_state_encoded(state, action, player)
                else:
                    action = random.choice(gopher.get_valid_moves(state))
                    state = gopher.get_next_state_idx(state, action, player)
            else:
                if mcts_first:
                    action = random.choice(gopher.get_valid_moves(state))
                    state = gopher.get_next_state_idx(state, action, player)
                else:
                    neutral_state = gopher.change_perspective(state, -1)
                    mcts_probs = mcts.search(neutral_state)
                    action = np.argmax(mcts_probs)
                    state = gopher.get_next_state_encoded(state, action, player)

            value, is_terminal = gopher.get_value_and_terminated(state, player)

            if is_terminal:
                gopher.display(state)
                print(f"Game {game + 1}: Game over! Player {player} wins!")
                if (mcts_first and player == 1) or (not mcts_first and player == -1):
                    mcts_wins += 1
                else:
                    rng_wins += 1
                break

            player = -player

    print(f"After {num_games} games:")
    print(f"MCTS wins: {mcts_wins}")
    print(f"RNG wins: {rng_wins}")
    

def main_dodo():
    dodo = DodoGame()

    args = {
        "C": 2,
        "num_searches": 600,
        "dirichlet_epsilon": 0.0,
        "dirichlet_alpha": 0.3,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model random pour essayer
    mock_resnet = MockResNet(dodo, num_resBlocks=5, num_hidden=128, device=device)

    mcts = MCTSDodo(game=dodo, args=args, model=mock_resnet)

    state = dodo.get_initial_state()

    player = 1
    while True:
        print(state)
        if player == 1:
            change_perspective = False
            actions = dodo.get_valid_moves_encoded(state, player, change_perspective)
            action = random.choice(np.where(actions)[0])
            state = dodo.get_next_state_encoded(state, action, player)
        else:
            change_perspective = True
            neutral_state = dodo.change_perspective(state, -1)
            actions = np.argmax(mcts.search(neutral_state, change_perspective))
            state = dodo.get_next_state_encoded(state, action, player)

        value, is_terminal = dodo.get_value_and_terminated(state, player)

        if is_terminal:
            dodo.display(state)
            print(f"Player {player} has won")
            break

        player = dodo.get_opponent(player)


if __name__ == "__main__":
    main_rng_gopher()
