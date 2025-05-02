from typing import List

from referee.game import Action, Coord, PlayerColor, Direction, MoveAction, GrowAction


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

    def min_max_value(self, curr_board: dict[Coord, str], color : PlayerColor, depth: int, maximizing_player : bool) -> int | List[Action] | Action:
        new_depth = depth - 1
        new_color = self.opposite_color(color)
        if self.terminal_test():
            # trace back the serious of actions
            return self.evaluation_function()
        if maximizing_player:
            max_value = int('-inf')
            #for each frog on the board
            value = self.min_max_value(self.apply_action(curr_board, GrowAction), new_color, new_depth, False)
            max_value = max(max_value, value)
            for frog in self.get_frog_coords(curr_board, color):
                #for each possible direction
                for direction in self.get_possible_directions(color): #possible moves should also include jumps
                    move = frog + direction
                    #if is a lily pad -> end
                    if self.is_valid_move(curr_board, move):
                        # apply the move (also removes the lily pad)
                        value = self.min_max_value(self.apply_action(curr_board, move), new_color, new_depth, False)
                        max_value = max(max_value, value)

                    #if the next location is a frog -> apply the move ->
                    elif self.is_valid_jump(curr_board, move):
                        # need to check every possible jumps
                        for jump in self.get_all_possible_jumps(move, curr_board , color):
                            value = self.min_max_value(self.apply_action(curr_board, jump), new_color, new_depth, False)
                            max_value = max(max_value, value)
            return max_value
        else:
            min_value = int('inf')
            #for each frog on the board
            value = self.min_max_value(self.apply_action(curr_board, GrowAction), new_color, new_depth, True)
            min_value = min(min_value, value)
            for frog in self.get_frog_coords(curr_board, color):
                #for each possible direction
                for direction in self.get_possible_directions(color): #possible moves should also include jumps
                    move = frog + direction
                    #if is a lily pad -> end
                    if self.is_valid_move(curr_board, move):
                        # apply the move (also removes the lily pad)
                        value = self.min_max_value(self.apply_action(curr_board, move), new_color, new_depth, True)
                        min_value = min(min_value, value)
                    #if the next location is a frog -> apply the move ->
                    elif self.is_valid_jump(curr_board, move):
                        # need to check every possible jumps
                        for jump in self.get_all_possible_jumps(move, curr_board , color):
                            value = self.min_max_value(self.apply_action(curr_board, jump), new_color, new_depth, True)
                            min_value = min(min_value, value)
            return min_value


    def is_valid_move(self, board: dict[Coord, str], move: Coord) -> bool:
        """
        Check if the move is valid
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
                return [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.DownLeft, Direction.DownRight]
            #if blue ->
            case PlayerColor.BLUE:
                return [Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.UpLeft, Direction.UPRight]
            #if none ->
            case _:
                return []
    def temp_jump_only_consider_one(self, color: PlayerColor) -> list[Direction]:
        pass

    def get_all_possible_jumps(self, curr: Coord, initial_board: dict[Coord, str], color : PlayerColor) -> list[Coord]:
        reachable_nodes = set()
        stack = [curr] # starting point
        while stack:
            exploring_node = stack.pop()
            if exploring_node in reachable_nodes:
                continue
            reachable_nodes.add(exploring_node)
            #explore all the neighbors
            for direction in self.get_possible_directions(color):
                #check if the next node can jump
                node_can_jump = exploring_node + direction #a frog
                if self.can_jump(initial_board, node_can_jump, direction): # if the neighbour is a frog
                    #move current location to the next node
                    landing_node = exploring_node + direction * 2
                    stack.append(landing_node)
                else:# where the jump sequence is ended - so no more
                    ...
        return list(reachable_nodes)

    def can_jump(self, cur_board : dict[Coord, str], neighbour_node, direction) -> bool:
        """
        Check if the frog can jump
        1. the neighbour_node node must be in the board
        2. the neighbour_node must be a frog
        """
        if neighbour_node not in cur_board or cur_board[neighbour_node] == 'l': #
            return False
        if cur_board[neighbour_node] == 'r' or cur_board[neighbour_node] == 'b': # is a frog
            #check the direction
            landing_node = neighbour_node + direction
            if neighbour_node not in cur_board or cur_board[landing_node] != 'l': #if not lily then it cannot land
                return False
            else: return True
        return False


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
