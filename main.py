import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gopher import *
from dodo import *
import time


def main_alpha_gopher():
    # Définir les arguments pour MCTS
    args = {
        "C": 2,
        "num_searches": 1200,
        "dirichlet_epsilon": 0.0,
        "dirichlet_alpha": 0.3,
    }
    # Initialiser le jeu
    gopher = GopherGame(board_size=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model random pour essayer
    model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
    model.load_state_dict(torch.load("model_1_GopherGame.pt", map_location=device))
    # Initialiser MCTS
    mcts = MCTSAlpha(game=gopher, args=args, model=model)

    # Obtenir l'état initial
    state = gopher.get_initial_state()
    player = 1

    # Commencer le chronomètre
    start_time = time.time()

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

            # Arrêter le chronomètre
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")

            break

        player = -player


def main_dodo():
    game = DodoGame2()
    state = game.init_board()
    current_player = 1  # Rouge commence
    turns = 0
    while True:
        game.display(state)

        if game.is_terminal_state(state, current_player):
            break

        if current_player == 1:  # Minimax player
            move = random.choice(game.get_valid_moves(state, current_player))
            print(move)
        else:
            move, evaluation = minimax_dodo(
                game,
                state,
                10,
                float("-inf"),
                float("inf"),
                True,
                current_player,
                game.memo,
            )
            print(game.get_valid_moves(state, current_player))
            print(move, evaluation)
        state = game.get_next_state(state, move[0], move[1], current_player)

        current_player = -current_player  # Change player
        turns += 1

    print(f"Game over in {turns} turns.")
    print(f"Gagnant: {current_player}")
    game.display(state)


def main_gopher_classique():
    game = GopherGame2()
    state = game.get_initial_state()
    current_player = 1  # Rouge commence
    turns = 0
    while True:
        game.display(state)

        if game.is_terminal_state(state, current_player):

            break

        if current_player == 1:
            move = minimax_gopher(
                game,
                state,
                10,
                float("-inf"),
                float("inf"),
                True,
                current_player,
            )[1]
            print(move)
            if move is None:
                move = random.choice(game.get_valid_moves(state, current_player))
        else:
            move = random.choice(game.get_valid_moves(state, current_player))
        state = game.get_next_state(state, move, current_player)

        current_player = -current_player  # Change player
        turns += 1

    print(f"Game over in {turns} turns.")
    print(f"Gagnant: {-current_player}")
    game.display(state)


if __name__ == "__main__":
    main_alpha_gopher()
