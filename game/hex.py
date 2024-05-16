import math
from collections import namedtuple

# Structures et fonctions utilitaires pour les hexagones
Point = namedtuple("Point", ["x", "y"])
Hex = namedtuple("Hex", ["q", "r", "s"])

def hex_add(a, b):
    return Hex(a.q + b.q, a.r + b.r, a.s + b.s)

def hex_subtract(a, b):
    return Hex(a.q - b.q, a.r - b.r, a.s - b.s)

def hex_neighbor(hex, direction):
    hex_directions = [Hex(1, 0, -1), Hex(1, -1, 0), Hex(0, -1, 1), Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1)]
    return hex_add(hex, hex_directions[direction])