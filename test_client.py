#!/usr/bin/python3

import ast
import argparse
from typing import Dict, Any
from gndclient import start, Action, Score, Player, State, Time, DODO_STR, GOPHER_STR
import random
import torch
import numpy as np
from gopher import *
from dodo import *

Environment = Dict[str, Any]


def initialize(
    game: str, state: State, player: Player, hex_size: int, total_time: Time
) -> Environment:
    print("Init")
    print(
        f"{game} playing {player} on a grid of size {hex_size}. Time remaining: {total_time}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if game == GOPHER_STR:
        game_instance = GopherGame(board_size=hex_size)
        if hex_size == 6:
            model = ResNet(
                game_instance, num_resBlocks=9, num_hidden=256, device=device
            )
            model.load_state_dict(
                torch.load("model_0_GopherGame.pt", map_location=device)
            )
        else:
            model = MockResNet(
                game_instance, num_resBlocks=5, num_hidden=128, device=device
            )
        mcts = MCTSAlpha(
            game=game_instance,
            args={
                "C": 2,
                "num_searches": 1200,
                "dirichlet_epsilon": 0.0,
                "dirichlet_alpha": 0.3,
            },
            model=model,
        )
    else:
        game_instance = DodoGame()
        model = MockResNet(
            game_instance, num_resBlocks=5, num_hidden=128, device=device
        )
        mcts = MCTSDodo(
            game=game_instance,
            args={
                "C": 2,
                "num_searches": 600,
                "dirichlet_epsilon": 0.0,
                "dirichlet_alpha": 0.3,
            },
            model=model,
        )

    return {
        "game": game_instance,
        "model": model,
        "mcts": mcts,
        "device": device,
        "type": game,
    }


def strategy_brain(
    env: Environment, state: State, player: Player, time_left: Time
) -> tuple[Environment, Action]:
    print("New state ", state)
    print("Time remaining ", time_left)

    game = env["game"]
    mcts = env["mcts"]

    if env["type"] == GOPHER_STR:
        state = game.serveur_state_to_gopher(state)
        if player == 2:
            player = -1
        game.display(state)
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action_encoded = np.argmax(mcts_probs)
        state = game.get_next_state_encoded(state, action_encoded, player)
        action = game.encoded_to_server(action_encoded)

    elif env["type"] == DODO_STR:
        state = game.serveur_state_to_gopher(state)
        if player == 2:
            player = -1
        change_perspective = player != 1
        neutral_state = game.change_perspective(state, player, change_perspective)
        mcts_probs = mcts.search(neutral_state, change_perspective)
        action_encoded = np.argmax(mcts_probs)
        action = dodo.decode_action_serveur(action_encoded)

    return (env, action)


def final_result(state: State, score: Score, player: Player):
    print(f"Ending: {player} wins with a score of {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ClientTesting", description="Test the IA02 python client"
    )

    parser.add_argument("group_id")
    parser.add_argument("members")
    parser.add_argument("password")
    parser.add_argument("-s", "--server-url", default="http://localhost:8080/")
    parser.add_argument("-d", "--disable-dodo", action="store_true")
    parser.add_argument("-g", "--disable-gopher", action="store_true")
    args = parser.parse_args()

    available_games = [DODO_STR, GOPHER_STR]
    if args.disable_dodo:
        available_games.remove(DODO_STR)
    if args.disable_gopher:
        available_games.remove(GOPHER_STR)

    start(
        args.server_url,
        args.group_id,
        args.members,
        args.password,
        available_games,
        initialize,
        strategy_brain,
        final_result,
        gui=True,
    )
