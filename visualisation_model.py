import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gopher import *

# Déterminer l'appareil à utiliser (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialiser le jeu avec une taille de plateau de 6
gopher = GopherGame(board_size=6)

# Calculer la taille totale du plateau
board_size = 2 * gopher.size + 1

# Obtenir l'état initial du jeu
state = gopher.get_initial_state()
# Mettre à jour l'état avec un coup initial
state = gopher.get_next_state(state, (0, -4, 4), 1)

# Afficher l'état actuel du jeu
gopher.display(state)

# Encoder l'état actuel pour le modèle
encoded_state = gopher.get_encoded_state(state)

# Convertir l'état encodé en tenseur PyTorch et ajouter une dimension pour le batch
tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

# Initialiser le modèle ResNet avec les paramètres spécifiés
model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
# Charger les poids du modèle entraîné
model.load_state_dict(torch.load("model_1_GopherGame.pt", map_location=device))
# Mettre le modèle en mode évaluation
model.eval()

# Passer l'état à travers le modèle pour obtenir les prédictions de la politique et de la valeur
policy, value = model(tensor_state)
# Extraire la valeur prédite et la convertir en nombre
value = value.item()
# Appliquer la fonction softmax aux prédictions de la politique pour obtenir des probabilités
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
# Masquer les coups invalides dans la politique
policy = policy * gopher.get_valid_moves_encoded(state)

# Reshaper les valeurs de la politique en une grille correspondant au plateau de jeu
policy_grid = policy.reshape(board_size, board_size)

# Création des indices pour les barres 3D
x = np.arange(board_size)
y = np.arange(board_size)
x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)

# Hauteur des barres (valeurs de la politique)
dx = dy = 0.5
dz = policy_grid.flatten()

# Création de la figure et de l'axe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Création des barres en 3D pour visualiser les valeurs de la politique sur le plateau
ax.bar3d(x, y, z, dx, dy, dz, shade=True)

# Ajout des étiquettes et du titre au graphique
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Policy Value")
ax.set_title("3D Bar Plot of Game Board Policies")

# Affichage du graphique
plt.show()