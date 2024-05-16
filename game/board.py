from game.hex import Hex, Point, hex_neighbor, hex_add, hex_subtract

class Board_gopher:
    def __init__(self, size=6):
        self.size = size-1
        self.grid = {}  # Utiliser un dictionnaire pour stocker les hexagones et leurs états

    def place_stone(self, q, r, s, color):
        hex = Hex(q, r, s)
        if hex not in self.grid:
            self.grid[hex] = color
            return True
        return False
    
    #verifier si le mouvement est valide donc case vide + dans les limites du plateau
    def is_valid_move(self, q, r, s):
        if Hex(q, r, s) in self.grid or abs(q) > self.size or abs(r) > self.size or abs(s) > self.size:
            return False
        return True

    def display(self):
        # Affichage texte du plateau
        for q in range(-self.size, self.size + 1):
            # Calculer l'indentation pour chaque ligne
            indent = abs(q)
            print(' ' * indent, end='')
            for r in range(-self.size, self.size + 1):
                s = -r - q
                if abs(r) <= self.size and abs(q) <= self.size and abs(s) <= self.size:
                    hex = Hex(q, r, s)
                    if hex in self.grid:
                        print(self.grid[hex], end=' ')
                    else:
                        print('.', end=' ')
            print()

    def has_winner(self):
        # Logique pour déterminer s'il y a un gagnant
        pass