from game.game_logic import GopherGame
from game.player import Player, AIPlayer
from game.hex import oddr_to_axial, axial_to_oddr

def initialize(game: str, state: State, player: Player, hex_size: int, total_time: Time) -> Environment:
    # Initialiser le plateau de jeu
    board = Board_gopher(size=hex_size)
    players = [AIPlayer(), AIPlayer()]
    players[0].color = 'R' if player == 1 else 'B'
    players[1].color = 'B' if player == 1 else 'R'
    
    # Initialiser l'état du jeu avec les pierres déjà placées
    for cell, pl in state:
        hex = oddr_to_axial(cell)
        color = 'R' if pl == 1 else 'B'
        board.place_stone(hex.q, hex.r, hex.s, color)
    
    game_instance = GopherGame(players[0], players[1], board_size=hex_size)
    environment = {
        'game': game_instance,
        'player': player,
        'total_time': total_time,
        'hex_size': hex_size
    }
    return environment

def strategy(env: Environment, state: State, player: Player, time_left: Time) -> tuple[Environment, ActionGopher]:
    game_instance = env['game']
    
    # Mettre à jour l'état du jeu avec les mouvements du joueur adverse
    for cell, pl in state:
        hex = oddr_to_axial(cell)
        color = 'R' if pl == 1 else 'B'
        if game_instance.board.is_valid_move(hex.q, hex.r, hex.s):
            game_instance.board.place_stone(hex.q, hex.r, hex.s, color)
    
    # Utiliser la stratégie de l'IA pour déterminer le meilleur coup
    current_player = game_instance.get_current_player()
    move = current_player.strategy(game_instance)
    q, r, s = move
    
    # Placer la pierre sur le plateau
    game_instance.board.place_stone(q, r, s, current_player.color)
    
    # Convertir les coordonnées axiales en coordonnées offset pour le retour
    offset_move = axial_to_oddr(Hex(q, r, s))
    
    # Mettre à jour l'environnement
    env['game'] = game_instance
    
    return env, offset_move

def final_result(state: State, score: Score, player: Player):
    print(f"Player {player} finished with score {score}.")
    print("Final state:")
    for cell, pl in state:
        print(f"Cell: {cell}, Player: {pl}")

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