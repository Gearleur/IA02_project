from game.game_logic import GopherGame
from game.player import Player, AIPlayer

def main():
    # Crée les joueurs
    player1 = Player('R')  # Rouge
    player2 = AIPlayer('B')  # Bleu (IA)
    
    # Crée le jeu avec les joueurs passés en paramètre
    game = GopherGame(player1, player2, board_size=6)
    
    while True:
        game.board.display()
        if game.play_turn():
            break

if __name__ == "__main__":
    main()