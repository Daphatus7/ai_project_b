from typing import List

from referee.game import Action, Coord, PlayerColor, Direction, MoveAction, GrowAction
from .program import print_board

class MinMaxSearch:
    """
    MinMaxSearch class for implementing the MinMax algorithm for game AI.
    """

    def __init__(self, board: dict[Coord, str], depth, color: PlayerColor):
        """
        The initial state of the board
        """
        self.board = board
        self.depth = depth
        self.color = color
        self.best_move = None
        self.cached_states = {}

    def min_max_value(self, board: dict[Coord, str], color : PlayerColor, depth: int, maximizing_player : bool, action : []) -> int | List[Action] | Action:
        if self.terminal_test():
            # trace back the serious of actions
            return self.evaluation_function()
        if maximizing_player:
            max_value = int('-inf')
            #for each frog on the board
            for frog in self.get_frog_coords(board, color):
                #for each possible direction
                for direction in self.get_possible_directions(color): #possible moves should also include jumps
                    move = frog + direction
                    #if is a lily pad -> end
                    if self.is_valid_move(board, move):
                        # apply the move (also removes the lily pad)
                        ...
                    #if the next location is a frog -> apply the move ->
                    elif self.is_valid_jump(board, move):
                        # need to check every possible jumps
                        for jump in self.get_all_possible_jumps(board, color):
                            # every possible destination
                            # apply the jump
                            ...
        else:
            min_value = int('inf')
            for direction in self.get_possible_directions(color):
                if self.is_valid_move(board, direction):
                    value = self.min_max_value(self.apply_action(board, direction), self.opposite_color(color), depth - 1, True)
                    min_value = min(min_value, value)
            return min_value

    def is_valid_move(self, board: dict[Coord, str], move: Coord) -> bool:
        """
        Check if the move location has a lily pad
        """
        return move in board and board[move] == 'l'
    
    def is_valid_jump(self, board: dict[Coord, str], move: Coord) -> bool:
        return
    
    def terminal_test(self, board=None) -> bool:
        """
        Check if the game is over, i.e., if all frogs of one color are on the opposite side of the board.
        """
        
        # Check if all RED frogs are at row 7
        red_frogs = [coord for coord, state in board.items() if state == 'r']
        if all(frog.r == 7 for frog in red_frogs) and red_frogs:
            return True
            
        # Check if all BLUE frogs are at row 0
        blue_frogs = [coord for coord, state in board.items() if state == 'b']
        if all(frog.r == 0 for frog in blue_frogs) and blue_frogs:
            return True
            
        return False

    def apply_action(self, board: dict[Coord, str], action: Action) -> dict[Coord, str]:
        """
        Apply the move to the board and return the new board state.
        """

        # If it is a move action
        if isinstance(action, MoveAction):
            # Start coordinate of the frog
            start_coord = action.coord
            current_pos = start_coord
            # Sequence of directions to move
            directions = action.directions
            # Keep the color of the frog
            frog_color = board[start_coord]
            # Remove the frog and the state of the cell
            del board[start_coord]
            # Apply moves
            for direction in directions:
                current_pos += direction

            board[current_pos] = frog_color

        # If it is a grow action
        elif isinstance(action, GrowAction):
            player_color = getattr(action, 'player_color', self.color)
            char = 'r' if player_color == PlayerColor.RED else 'b'
            frogs = [coord for coord, state in board.items() if state == char]

            # For each frog add lily pads adjacent to it
            for frog in frogs:
                for dir_r in range(-1, 2):
                    for dir_c in range(-1, 2):
                        # Skip if at frog's position
                        if dir_r == 0 and dir_c == 0:
                            continue

                        new_coord = frog + Coord(dir_r, dir_c)
                        if 0 <= new_coord.r < 8 and 0 <= new_coord.c < 8: # Within bounds
                            # If the new coordinate is empty, add a lily pad
                            if new_coord not in board:
                                board[new_coord] = 'l'

        return board

    def evaluation_function(self) -> int:
        return int('inf')

    def opposite_color(self, color: PlayerColor) -> PlayerColor|None:
        """
        Get the opposite color of the player
        """
        match color:
            case PlayerColor.RED:
                return PlayerColor.BLUE
            case PlayerColor.BLUE:
                return PlayerColor.RED
            case _:
                return None

    def get_frog_coords(self, board: dict[Coord, str], color: PlayerColor) -> list[Coord]:
        """
        Get the frogs of the player
        """
        frogs = []
        for coord, state in board.items():
            if state.state == color:
                frogs.append(coord)
        return frogs

    def get_possible_directions(self, color : PlayerColor)-> list[Direction]:
        """
        Get the possible moves for the player
        """
        #if red ->
        match color:
            case PlayerColor.RED:
                return [Direction.Right, Direction.Left, Direction.Down, Direction.DownLeft, Direction.DownRight]
            #if blue ->
            case PlayerColor.BLUE:
                return [Direction.Right, Direction.Left, Direction.Up, Direction.UpLeft, Direction.UpRight]
            #if none ->
            case _:
                return []
            
    def temp_jump_only_consider_one(self, color: PlayerColor) -> list[Direction]:
        pass

    def get_all_possible_jumps(self, board: dict[Coord, str], color: PlayerColor) -> list[Action]:
        """
        Get all possible jumps for the player
        -returns a series of jumps
        """
        jumps = []

        return jumps
    
    def bfs(self, start: Coord, end: Coord) -> list[Coord]:
        return []

    def get_best_move(self) -> Action:
        """
        Find the best move using the MinMax algorithm
        """
        best_value = int('-inf')
        best_move = None

        for move in self.get_possible_moves():
            if self.is_valid_move(self.board, move):
                new_board = self.apply_action(self.board, move)
                value = self.min_max_value(new_board, self.depth - 1, False)

                if value > best_value:
                    best_value = value
                    best_move = move

        self.best_move = best_move

        return best_move
