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
    hex_directions = [
        Hex(1, 0, -1),
        Hex(1, -1, 0),
        Hex(0, -1, 1),
        Hex(-1, 0, 1),
        Hex(-1, 1, 0),
        Hex(0, 1, -1),
    ]
    return hex_add(hex, hex_directions[direction])


def axial_to_oddr(hex):
    col = hex.q
    row = hex.r + (hex.q - (hex.q & 1)) // 2
    return Point(col, row)


def oddr_to_axial(point):
    q = point.x
    r = point.y - (point.x - (point.x & 1)) // 2
    return Hex(q, r)


def idx_to_hex(x, y, board_size):
    return Hex(y - board_size, x - board_size, -(x - board_size) - (y - board_size))


def hex_to_idx(hex, board_size):
    return hex.r + board_size, hex.q + board_size
