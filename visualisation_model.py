import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gopher import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialiser le jeu
gopher = GopherGame(board_size=6)

state = gopher.get_initial_state()
moves = gopher.get_valid_moves_encoded(state)
state = gopher.get_next_state(state, (-3, 0, 3), 1)
actions = gopher.get_valid_moves_encoded(state)
print(actions)

encoded_state = gopher.get_encoded_state(state)
print(encoded_state)

tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
model.load_state_dict(torch.load("model_0_GopherGame.pt", map_location=device))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value)
print(policy)

gopher.display(state)
print(tensor_state)

plt.bar(range(gopher.action_size), policy * gopher.get_valid_moves_encoded(state))

# prendre le argmin de la policy différent de 0
action = policy * gopher.get_valid_moves_encoded(state)

action = np.argmax(action)


state = gopher.get_next_state_encoded(state, action, -1)
gopher.display(state)

plt.show()
