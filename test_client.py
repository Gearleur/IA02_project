#!/usr/bin/python3

import ast
import argparse
import random
from typing import Dict, Any, Tuple
from gndclient import start, Action, Score, Player, State, Time, DODO_STR, GOPHER_STR
from dodo import *
from gopher import *

Environment = Dict[str, Any]

def initialize(game: str, state: State, player: Player, hex_size: int, total_time: Time) -> Environment:
    env = {
        "game": DodoGame2() if game == DODO_STR else GopherGame2(),
        "state": state,
        "current_player": 1 if player == "RED" else -1,
        "memo_keys": [],
        "memo_vals": []
    }

    if game == GOPHER_STR:
        args = {
            "C": 2,
            "num_searches": 1200,
            "dirichlet_epsilon": 0.0,
            "dirichlet_alpha": 0.3,
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNet(env["game"], num_resBlocks=9, num_hidden=256, device=device)
        model.load_state_dict(torch.load("model_1_GopherGame.pt", map_location=device))
        env["mcts"] = MCTSAlpha(game=env["game"], args=args, model=model)

    return env

def strategy_brain(env: Environment, state: State, player: Player, time_left: Time) -> Tuple[Environment, Action]:
    game = env["game"]
    current_player = env["current_player"]

    if isinstance(game, DodoGame2):
        state = game.server_state_to_dodo(state)
        move, _ = minimax_dodo(
            game, state, 10, float("-inf"), float("inf"), True, current_player, game.memo
        )
        print(move)
        start, end = move
        new_env = {
            "game": game,
            "state": game.get_next_state(state, start, end, current_player),
            "current_player": -current_player,
        }
        
        move = game.action_to_server(move)
        print(move)
    else:
        print(state)
        state = game.serveur_state_to_gopher(state)
        print(state)
        move = minimax_gopher(
                game,
                state,
                10,
                float("-inf"),
                float("inf"),
                True,
                current_player,
                game.memo,
        )[1]
        move = game.action_to_server(move)

        new_env = {
            "game": game,
            "state": game.get_next_state(state, move, current_player),
            "current_player": -current_player,
        }
        
    return new_env, move


def final_result(state: State, score: Score, player: Player):
    print(f"Ending: {player} wins with a score of {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ClientTesting", description="Test the IA02 python client")

    parser.add_argument("group_id")
    parser.add_argument("members")
    parser.add_argument("password")
    parser.add_argument("-s", "--server-url", default="http://localhost:8000")
    parser.add_argument("-d", "--disable-dodo", action="store_true")
    parser.add_argument("-g", "--disable-gopher", action="store_true")
    args = parser.parse_args()

    available_games = [DODO_STR, GOPHER_STR]
    if args.disable_dodo:
        available_games.remove(DODO_STR)
    if args.disable_gopher:
        available_games.remove(GOPHER_STR)

    start(args.server_url, args.group_id, args.members, args.password, available_games, initialize, strategy_brain, final_result, gui=True)
