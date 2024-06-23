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

board_size = 2*gopher.size +1

state = gopher.get_initial_state()
state = gopher.get_next_state(state, (0, -4, 4), 1)

gopher.display(state)

encoded_state = gopher.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
model.load_state_dict(torch.load("model_1_GopherGame.pt", map_location=device))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
policy = policy * gopher.get_valid_moves_encoded(state)

policy_grid = policy.reshape(board_size, board_size)

# Création des indices pour les barres
x = np.arange(board_size)
y = np.arange(board_size)
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)

# Hauteur des barres
dx = dy = 0.5
dz = policy_grid.flatten()

# Création de la figure et de l'axe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Création des barres en 3D
ax.bar3d(x, y, z, dx, dy, dz, shade=True)

# Ajout des étiquettes et du titre
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Policy Value')
ax.set_title('3D Bar Plot of Game Board Policies')

# Affichage du graphique
plt.show()