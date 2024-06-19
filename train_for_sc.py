import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from gopher import *

# Spécifiez les GPUs à utiliser
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Utiliser les GPUs 0, 1 et 2

gopher = GopherGame(board_size=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)

# Utiliser DataParallel pour utiliser plusieurs GPUs
if torch.cuda.device_count() > 1:
    print(f"Utilisation de {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args = {
    'num_searches': 600,
    'C': 2,
    'num_iterations': 6,
    'num_selfPlay_iterations': 150,
    'num_parallel_games': 10,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
}

alpha_zero = AlphaZeroParallel(model, optimizer, gopher, args)
alpha_zero.learn()