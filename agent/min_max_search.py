from referee.game import Action, Coord, PlayerColor, Direction
from referee.game.board import CellState
from referee.game.actions import MoveAction, GrowAction


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
        self.color = color
        self.best_move = None
        self.cached_states = {}


    def min_max_value(self, board: dict[Coord, CellState], depth: int, maximizing_player : bool) -> int:
        if self.terminal_test():
            return self.evaluation_function()
        if maximizing_player:
            max_value = int('-inf')
            for move in self.get_possible_moves():
                if self.is_valid_move(board, move):
                    value = self.min_max_value(self.apply_move(board, move), depth - 1, False)
                    max_value = max(max_value, value)
            return max_value
        else :
            min_value = int('inf')
            for move in self.get_possible_moves():
                if self.is_valid_move(board, move):
                    new_board = self.apply_move(board, move)
                    value = self.min_max_value(new_board ,depth - 1, True)
                    min_value = min(min_value, value)
            return min_value
        

    def is_valid_move(self, board: dict[Coord, CellState], move: Action) -> bool:
        """
        Check if the move is valid
        """
        if isinstance(move, MoveAction):
            coord = move.coord
            directions = move.directions

        not_valid_directions = [Direction.Up, Direction.UpLeft, Direction.UpRight]

        for direction in directions:
            if direction in not_valid_directions:
                return False          

        return True
    

    def terminal_test(self, board: dict[Coord, CellState]) -> bool:
        """
        Check if the current board state is a terminal state
        """

        # Check if either player has their frogs on the opposite side of the board
        for coord, cell in board.items():
            if cell.state == PlayerColor.RED and coord.row == len(board) - 1:
                return True
            if cell.state == PlayerColor.BLUE and coord.row == 0:
                return True
        
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


    def get_best_move(self) -> Action:
        """
        Find the best move using the MinMax algorithm
        """
        best_value = int('-inf')
        best_move = None

        for move in self.get_possible_moves():
            if self.is_valid_move(self.board, move):
                new_board = self.apply_move(self.board, move)
                value = self.min_max_value(new_board, self.depth - 1, False)

                if value > best_value:
                    best_value = value
                    best_move = move
                    
        self.best_move = best_move

        return best_move