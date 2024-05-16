from game.game_logic import GopherGame
from game.player import Player, AIPlayer

def main():
    # Crée les joueurs
    player1 = Player()
    player2 = AIPlayer()
    player3 = AIPlayer()
    
    # Crée le jeu avec les joueurs passés en paramètre
    game = GopherGame(player2, player3, board_size=6)
    
    while True:
        if game.play_turn():
            break

if __name__ == "__main__":
    main()