import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from gopher import *

gopher = GopherGame(board_size=6)

device = torch.device("cuda")

model = ResNet(gopher, num_resBlocks=9, num_hidden=128, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

model.load_state_dict(torch.load("model_2_GopherGame.pt", map_location=device))
optimizer.load_state_dict(torch.load("optimizer_2_GopherGame.pt", map_location=device))

args = {
    "num_searches": 800,
    "C": 2,
    "num_iterations": 8,
    "num_selfPlay_iterations": 160,
    "num_parallel_games": 40,
    "num_epochs": 4,
    "batch_size": 128,
    "temperature": 1.25,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.3,
}

alpha_zero = AlphaZeroParallel(model, optimizer, gopher, args)
alpha_zero.learn()
