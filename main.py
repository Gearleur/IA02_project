from game.game_logic import GopherGame, DodoGame
from game.player import Player, AIPlayer
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
    # Crée les joueurs
    player1 = Player()
    player2 = Player()
    # Crée le jeu avec les joueurs passés en paramètre
    game = GopherGame_2(player1, player2, board_size=6)
    
    while True:
        print(game.grid)
        if game.play_turn():
            break
        
def main_ia():
    # Nombre de parties à jouer
    num_games = 200
    
    # Scores des joueurs
    ai_wins = 0
    random_wins = 0

    for _ in range(num_games):
        # Crée les joueurs
        random_player = RandomPlayer()
        ai_player = AIPlayer(depth=3)  # Vous pouvez ajuster la profondeur selon vos besoins
        
        # Décide aléatoirement qui commence
        if random.choice([True, False]):
            player1, player2 = random_player, ai_player
        else:
            player1, player2 = ai_player, random_player

        # Crée le jeu avec les joueurs passés en paramètre
        game = GopherGame(player1, player2, board_size=6)
        
        while True:
            if game.play_turn():
                # Déterminer le gagnant
                winner = game.get_current_player().color
                if winner == ai_player.color:
                    ai_wins += 1
                else:
                    random_wins += 1
                break

    # Afficher les résultats
    print(f"AI Wins: {ai_wins}")
    print(f"Random Wins: {random_wins}")


def main_dodo():
    game = DodoGame(4)
    game.display()

if __name__ == "__main__":
    main_gopher()