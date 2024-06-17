import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from game import *

gopher = GopherGame(board_size=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args = {
    'num_searches': 800,
    'C': 2,
    'num_iterations': 10,
    'num_selfPlay_iterations': 800,
    'num_parallel_games': 800,
    'num_epochs': 6,
    'batch_size': 256,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
}

alpha_zero = AlphaZeroParallel(model, optimizer, gopher, args)
alpha_zero.learn()