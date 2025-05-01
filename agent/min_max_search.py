from referee.game import Action, Coord, PlayerColor, Direction


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

    def min_max_value(self, board: dict[Coord, str], color : PlayerColor, depth: int, maximizing_player : bool) -> int:
        if self.terminal_test():
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
                    value = self.min_max_value(self.apply_move(board, direction), self.opposite_color(color), depth - 1, True)
                    min_value = min(min_value, value)
            return min_value

    def is_valid_move(self, board: dict[Coord, str], move: Coord) -> bool:
        """
        Check if the move is valid
        """
        return move in board and board[move] == 'l'
    def is_valid_jump(self, board: dict[Coord, str], move: Coord) -> bool:
        return
    def terminal_test(self) -> bool:
        return False
    def apply_move(self,board: dict[Coord, str], move: Action) -> dict[Coord, str]:
        """
        Apply the move to the board
        """
        pass

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
        for coord, cell in board.items():
            if cell.state == color:
                frogs.append(coord)
        return frogs


    def get_possible_directions(self, color : PlayerColor)-> list[Direction]:
        """
        Get the possible moves for the player
        """
        #if red ->
        match color:
            case PlayerColor.RED:
                return [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.DownLeft, Direction.DownRight]
            #if blue ->
            case PlayerColor.BLUE:
                return [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.UpLeft, Direction.UPRight]
            #if none ->
            case _:
                return []
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
                new_board = self.apply_move(self.board, move)
                value = self.min_max_value(new_board, self.depth - 1, False)

                if value > best_value:
                    best_value = value
                    best_move = move

        self.best_move = best_move

        return best_move