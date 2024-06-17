import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gopher import *

gopher = GopherGame(board_size=6)

state = gopher.get_initial_state()

player = 1

while True:
    gopher.display(state)
    if player == 1:
        action = random.choice(gopher.get_valid_moves(state))
    else:
        action = random.choice(gopher.get_valid_moves(state))

    state = gopher.get_next_state_idx(state, action, player)
    
    value, is_terminal = gopher.get_value_and_terminated(state, action)
    
    if is_terminal:
        gopher.display(state)
        print(f"Game over! Player {-player} wins!")
        break
    player = gopher.get_opponent(player)