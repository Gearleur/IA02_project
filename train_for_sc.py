import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from gopher import *

# Spécifiez les GPUs à utiliser
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Utiliser les GPUs 0, 1 et 2

gopher = GopherGame()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetGPU(gopher, num_resBlocks=9, num_hidden=128, device=device)

# Utiliser DataParallel pour utiliser plusieurs GPUs
# if torch.cuda.device_count() > 1:
#     print(f"Utilisation de {torch.cuda.device_count()} GPUs")
#     model = nn.DataParallel(model)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

args = {
    "num_searches": 800,
    "C": 2,
    "num_iterations": 7,
    "num_selfPlay_iterations": 500,
    "num_parallel_games": 100,
    "num_epochs": 4,
    "batch_size": 32,
    "temperature": 1,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.3,
}

alpha_zerogpu = AlphaZeroParallelGPU(model, optimizer, gopher, args)
alpha_zerogpu.learn()
