import numpy as np
from game.hex import Hex, Point, hex_add, hex_subtract, hex_neighbor, idx_to_hex, hex_to_idx
import torch

#verifier cuda avec gpu
print(torch.cuda.is_available())