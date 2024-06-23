import math
from collections import namedtuple
from typing import Dict, Tuple, Optional, List

# Structures et fonctions utilitaires pour les hexagones
Point = namedtuple("Point", ["x", "y"])
Hex = namedtuple("Hex", ["q", "r", "s"])


def hex_add(a: Hex, b: Hex) -> Hex:
    """
    Ajoute deux hexagones.

    :param a: Premier hexagone
    :param b: Deuxième hexagone
    :return: Somme des deux hexagones
    """
    return Hex(a.q + b.q, a.r + b.r, a.s + b.s)


def hex_subtract(a: Hex, b: Hex) -> Hex:
    """
    Soustrait un hexagone d'un autre.

    :param a: Premier hexagone
    :param b: Deuxième hexagone
    :return: Différence des deux hexagones
    """
    return Hex(a.q - b.q, a.r - b.r, a.s - b.s)


def hex_neighbor(hex: Hex, direction: int) -> Hex:
    """
    Trouve un hexagone voisin dans une direction donnée.

    :param hex: Hexagone de départ
    :param direction: Direction (0 à 5)
    :return: Hexagone voisin dans la direction donnée
    """
    hex_directions = [
        Hex(1, 0, -1),
        Hex(1, -1, 0),
        Hex(0, -1, 1),
        Hex(-1, 0, 1),
        Hex(-1, 1, 0),
        Hex(0, 1, -1),
    ]
    return hex_add(hex, hex_directions[direction])


def axial_to_oddr(hex: Hex) -> Point:
    """
    Convertit des coordonnées axiales en coordonnées décalées (odd-r).

    :param hex: Hexagone en coordonnées axiales
    :return: Point en coordonnées décalées (odd-r)
    """
    col = hex.q
    row = hex.r + (hex.q - (hex.q & 1)) // 2
    return Point(col, row)


def oddr_to_axial(point: Point) -> Hex:
    """
    Convertit des coordonnées décalées (odd-r) en coordonnées axiales.

    :param point: Point en coordonnées décalées (odd-r)
    :return: Hexagone en coordonnées axiales
    """
    q = point.x
    r = point.y - (point.x - (point.x & 1)) // 2
    return Hex(q, r, -q - r)


def idx_to_hex(x: int, y: int, board_size: int) -> Hex:
    """
    Convertit des indices de tableau en coordonnées d'hexagone.

    :param x: Indice de colonne
    :param y: Indice de ligne
    :param board_size: Taille du plateau
    :return: Hexagone correspondant aux indices
    """
    return Hex(y - board_size, x - board_size, -(x - board_size) - (y - board_size))


def hex_to_idx(hex: Hex, board_size: int) -> tuple:
    """
    Convertit des coordonnées d'hexagone en indices de tableau.

    :param hex: Hexagone
    :param board_size: Taille du plateau
    :return: Indices de colonne et de ligne
    """
    return hex.r + board_size, hex.q + board_size

def rotate_hex(hex: Hex, angle: int) -> Hex:
    """Tourne les coordonnées d'un hexagone par un multiple de 60 degrés."""
    if angle % 60 != 0:
        raise ValueError("L'angle doit être un multiple de 60 degrés.")
    
    times = angle // 60 % 6
    if times == 1:
        return Hex(-hex.r, -hex.s, -hex.q)
    elif times == 2:
        return Hex(hex.s, hex.q, hex.r)
    elif times == 3:
        return Hex(-hex.q, -hex.r, -hex.s)
    elif times == 4:
        return Hex(hex.r, hex.s, hex.q)
    elif times == 5:
        return Hex(-hex.s, -hex.q, -hex.r)
    else:
        return hex  # times == 0, no rotation
