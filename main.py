from game.game_logic import GopherGame, DodoGame
from game.player import Player, AIPlayer, RandomPlayer
from game.hex import oddr_to_axial, axial_to_oddr
import random

# def initialize(game: str, state: State, player: Player, hex_size: int, total_time: Time) -> Environment:
#     # Initialiser le plateau de jeu
#     board = Board_gopher(size=hex_size)
#     players = [AIPlayer(), AIPlayer()]
#     players[0].color = 'R' if player == 1 else 'B'
#     players[1].color = 'B' if player == 1 else 'R'
    
#     # Initialiser l'état du jeu avec les pierres déjà placées
#     for cell, pl in state:
#         hex = oddr_to_axial(cell)
#         color = 'R' if pl == 1 else 'B'
#         board.place_stone(hex.q, hex.r, hex.s, color)
    
#     game_instance = GopherGame(players[0], players[1], board_size=hex_size)
#     environment = {
#         'game': game_instance,
#         'player': player,
#         'total_time': total_time,
#         'hex_size': hex_size
#     }
#     return environment

# def strategy(env: Environment, state: State, player: Player, time_left: Time) -> tuple[Environment, ActionGopher]:
#     game_instance = env['game']
    
#     # Mettre à jour l'état du jeu avec les mouvements du joueur adverse
#     for cell, pl in state:
#         hex = oddr_to_axial(cell)
#         color = 'R' if pl == 1 else 'B'
#         if game_instance.board.is_valid_move(hex.q, hex.r, hex.s):
#             game_instance.board.place_stone(hex.q, hex.r, hex.s, color)
    
#     # Utiliser la stratégie de l'IA pour déterminer le meilleur coup
#     current_player = game_instance.get_current_player()
#     move = current_player.strategy(game_instance)
#     q, r, s = move
    
#     # Placer la pierre sur le plateau
#     game_instance.board.place_stone(q, r, s, current_player.color)
    
#     # Convertir les coordonnées axiales en coordonnées offset pour le retour
#     offset_move = axial_to_oddr(Hex(q, r, s))
    
#     # Mettre à jour l'environnement
#     env['game'] = game_instance
    
#     return env, offset_move

# def final_result(state: State, score: Score, player: Player):
#     print(f"Player {player} finished with score {score}.")
#     print("Final state:")
#     for cell, pl in state:
#         print(f"Cell: {cell}, Player: {pl}")

def main_gopher():
    game = GopherGame()
    player1 = Player()
    player2 = Player()
    ai_player_1 = AIPlayer(game, depth=3)
    ai_player_2 = AIPlayer(game, depth=3)
    random_player = RandomPlayer()
    
    while not game.has_winner():
        game.display()
        if game.current_player == 1:
            q, r, s = ai_player_1.strategy(game)
        else:
            q, r, s = random_player.strategy(game)
        
        place_stone = game.place_stone(q, r, s, game.current_player)
        game.switch_player()
        
    game.display()
    print(f"Game over! Player {game.current_player} wins!")
        
def main_ia_gopher():
    num_games = 100
    ai_wins = 0
    random_wins = 0

    for _ in range(num_games):
        game = GopherGame()
        ai_player = AIPlayer(game, depth=3)
        random_player = RandomPlayer()

        # Tirer au sort celui qui commence
        if random.choice([True, False]):
            current_player = ai_player
            next_player = random_player
            game.current_player = 1  # AI Player starts
        else:
            current_player = random_player
            next_player = ai_player
            game.current_player = 2  # Random Player starts
        
        while not game.has_winner():
            if isinstance(current_player, AIPlayer):
                q, r, s = current_player.strategy(game)
            else:
                q, r, s = current_player.strategy(game)
            
            place_stone = game.place_stone(q, r, s, game.current_player)
            game.switch_player()

            # Alterner les joueurs
            current_player, next_player = next_player, current_player

        if game.current_player == 1:
            ai_wins += 1
        else:
            random_wins += 1

    print(f"AIPlayer won {ai_wins} times.")
    print(f"RandomPlayer won {random_wins} times.")


def main_dodo():
    game = DodoGame(4)
    game.display()

if __name__ == "__main__":
    main_ia_gopher()