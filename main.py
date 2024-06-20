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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialiser le jeu
    gopher = GopherGame(board_size=6)

    args = {
        "C": 2,
        "num_searches": 600,
        "dirichlet_epsilon": 0.0,
        "dirichlet_alpha": 0.3,
    }

    model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
    model.load_state_dict(torch.load("model_1_GopherGame.pt", map_location=device))
    model.eval()

    # Initialiser MCTS
    mcts = MCTSAlpha(game=gopher, args=args, model=model)

    state = gopher.get_initial_state()

    player = 1

    state = gopher.get_next_state(state, (-4, 0, 4), 1)
    neutral_state = gopher.change_perspective(state, -1)
    mcts_probs = mcts.search(neutral_state)
    print(mcts_probs)
    action = np.argmax(mcts_probs)
    state = gopher.get_next_state_encoded(state, action, -1)
    gopher.display(state)


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
            actions = np.where((mcts.search(neutral_state, change_perspective)))[0]
            for action in actions:
                print(dodo.decode_action(action))
            actions = mcts.search(neutral_state, change_perspective)
            print(actions)
            action = np.argmax(actions)
            state = dodo.get_next_state_encoded(state, action, player)
            # actions = dodo.get_valid_moves_encoded(neutral_state, 1, change_perspective=change_perspective)
            # # random move where action == 1
            # action = random.choice(np.where(actions)[0])
            # state = dodo.get_next_state_encoded(state, action, player)

        value, is_terminal = dodo.get_value_and_terminated(state, player)

        if is_terminal:
            dodo.display(state)
            print(f"Player {player} has won")
            break

        player = dodo.get_opponent(player)


if __name__ == "__main__":
    main_dodo()
