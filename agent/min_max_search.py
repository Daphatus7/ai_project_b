from referee.game import Action, Coord, PlayerColor
from referee.game.board import CellState


class MinMaxSearch:
    """
    MinMaxSearch class for implementing the MinMax algorithm for game AI.
    """

    def __init__(self, board: dict[Coord, CellState], depth, color: PlayerColor):
        """
        The initial state of the board
        """
        self.board = board
        self.depth = depth
        self.best_move = None
        self.cached_states = {}

    def min_max_value(self, board: dict[Coord, CellState], depth: int, maximizing_player : bool) -> int:
        if self.terminal_test():
            return self.evaluation_function()

        if maximizing_player:
            max_value = int('-inf')
            for frog in board:
                for move in self.get_possible_moves():
                    if self.is_valid_move(board, move):
                        value = self.min_max_value(self.apply_move(board, move), depth - 1, False)
                        max_value = max(max_value, value)
                        return max_value
        else:
            min_value = int('inf')
            for move in self.get_possible_moves():
                if self.is_valid_move(board, move):
                    value = self.min_max_value(self.apply_move(board, move),depth - 1, True)
                    min_value = min(min_value, value)
            return min_value

    def is_valid_move(self, board: dict[Coord, CellState], move: Action) -> bool:
        """
        Check if the move is valid
        """
        return True
    def terminal_test(self) -> bool:
        return False
    def apply_move(self,board: dict[Coord, CellState], move: Action) -> dict[Coord, CellState]:
        """
        Apply the move to the board
        """
        pass

    def evaluation_function(self)-> int:
        return int('inf')
    def get_possible_moves(self)-> list[Action]:
        pass