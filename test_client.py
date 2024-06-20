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
        gopher = GopherGame(board_size=hex_size)
        model = ResNet(gopher, num_resBlocks=9, num_hidden=256, device=device)
        model.load_state_dict(torch.load("model_0_GopherGame.pt", map_location=device))
        mcts = MCTSAlpha(game=gopher, args={"C": 2, "num_searches": 1800, "dirichlet_epsilon": 0.0, "dirichlet_alpha": 0.3}, model=model)
        return {"game": gopher, "model": model, "mcts": mcts, "device": device, "type": GOPHER_STR}
    elif game == DODO_STR:
        dodo = DodoGame()
        model = MockResNet(dodo, num_resBlocks=5, num_hidden=128, device=device)
        mcts = MCTSDodo(game=dodo, args={"C": 2, "num_searches": 1800, "dirichlet_epsilon": 0.0, "dirichlet_alpha": 0.3}, model=model)
        return {"game": dodo, "model": model, "mcts": mcts, "device": device, "type": DODO_STR}
    
def strategy_brain(
    env: Environment, state: State, player: Player, time_left: Time
) -> tuple[Environment, Action]:
    print("New state ", state)
    print("Time remaining ", time_left)
    
    game = env["game"]
    mcts = env["mcts"]

    if env["type"] == GOPHER_STR:
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action_encoded = np.argmax(mcts_probs)
        action = encoded_to_server(game, action_encoded, player)
        
    elif env["type"] == DODO_STR:
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