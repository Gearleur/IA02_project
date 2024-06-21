import random
import torch
import numpy as np
import math
from dodo import *
from gopher import *

def play_random_game():
    game = DodoGame2()
    state = game.initial_state
    player = 1  # 1 pour Rouge, -1 pour Bleu

    while True:
        valid_moves = game.get_valid_moves(state, player)

        if not valid_moves:
            winner = -player
            print(f"Player {winner} wins!")
            break

        start, end = random.choice(valid_moves)
        state = game.next_state(state, start, end)
        game.display(state)
        print()

        player = -player
        
play_random_game()