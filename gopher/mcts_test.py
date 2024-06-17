import random
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import logging

from .hex import Hex, Point, hex_neighbor, hex_add, hex_subtract
class mcts:
    def __init__(self, game, iterations=1000):
        self.game = game
        self.iterations = iterations
        self.wins = defaultdict(int)
        self.plays = defaultdict(int)


    def strategy(self, game):
        self.game = game
        self.max_player = game.current_player
        self.simulate()
        return self.get_best_move()

#execute plusieurs simulations de mcts
    def simulate(self):
        for _ in range(self.iterations):
            game_copy = self.clone_game(self.game)
            path = self.select(game_copy)
            winner = self.rollout(game_copy)
            self.backpropagate(path, winner)


#etape selection de l'algo
    def select(self, game):
        path = []
        while True:
            valid_moves = self.get_valid_moves(game)
            if not valid_moves:
                return path

            if all((move in self.plays) for move in valid_moves):
                log_total = math.log(sum(self.plays[move] for move in valid_moves))
                value, move = max(
                    ((self.wins[move] / self.plays[move]) + math.sqrt(2 * log_total / self.plays[move]), move)
                    for move in valid_moves
                )
            else:
                move = random.choice(valid_moves)

            path.append((game.current_player, move))
            game.place_stone(*move, game.current_player)
            game.switch_player()
            if game.has_winner():
                return path


    def rollout(self, game):
        while True:
            valid_moves = self.get_valid_moves(game)
            if not valid_moves:
                return 3 - game.current_player
            move = random.choice(valid_moves)
            game.place_stone(*move, game.current_player)
            game.switch_player()
            if game.has_winner():
                return 3 - game.current_player

    def backpropagate(self, path, winner):
        for player, move in path:
            if move not in self.plays:
                self.plays[move] = 0
                self.wins[move] = 0
            self.plays[move] += 1
            if player == winner:
                self.wins[move] += 1

    def get_best_move(self):
        move, _ = max(self.plays.items(), key=lambda item: item[1])
        return move

    def get_valid_moves(self, game):
        valid_moves = set()
        if not game.grid:
            # Premier tour, toutes les cases sont disponibles
            for q in range(-game.size, game.size + 1):
                for r in range(-game.size, game.size + 1):
                    s = -q - r
                    if abs(q) <= game.size and abs(r) <= game.size and abs(s) <= game.size:
                        valid_moves.add((q, r, s))
        else:
            for hex_pos in game.grid:
                for direction in range(6):
                    neighbor = hex_neighbor(hex_pos, direction)
                    if neighbor not in game.grid and abs(neighbor.q) <= game.size and abs(
                            neighbor.r) <= game.size and abs(neighbor.s) <= game.size:
                        has_enemy_connection = False
                        has_friendly_connection = False
                        for neighbor_dir in range(6):
                            adjacent = hex_neighbor(neighbor, neighbor_dir)
                            if adjacent in game.grid:
                                if game.grid[adjacent] == game.current_player:
                                    has_friendly_connection = True
                                else:
                                    has_enemy_connection = True
                        if has_enemy_connection and not has_friendly_connection:
                            valid_moves.add((neighbor.q, neighbor.r, neighbor.s))
        return list(valid_moves)

    def clone_game(self, game):
        new_game = GopherGame(game.size + 1)
        new_game.grid = game.grid.copy()
        new_game.current_player = game.current_player
        new_game.is_first_turn = game.is_first_turn
        return new_game