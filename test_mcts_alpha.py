import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from game import *

gopher = GopherGame(board_size=6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(gopher, num_resBlocks=4, num_hidden=64, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

args = {
    'num_searches': 600,
    'C': 2,
    'num_iterations': 3,
    'num_selfPlay_iterations': 5,
    'num_epochs': 5,
    'batch_size': 128,
    'temperature': 1,
}

alpha_zero = AlphaZero(model, optimizer, gopher, args)
alpha_zero.learn()