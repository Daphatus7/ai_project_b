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

    def min_max_value(self, curr_board: dict[Coord, str], color : PlayerColor, depth: int, maximizing_player : bool) -> float | List[Action] | Action:
        new_depth = depth - 1
        new_color = self.opposite_color(color)
        if self.terminal_test(curr_board, new_depth):
            return self.evaluation_function(curr_board, color)
        if maximizing_player:
            max_value = float('-inf')
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
                    elif self.can_jump(curr_board, move, direction):
                        # need to check every possible jumps
                        for jump in self.get_all_possible_jumps(move, curr_board , color):
                            value = self.min_max_value(self.apply_action(curr_board, jump), new_color, new_depth, False)
                            max_value = max(max_value, value)
            return max_value
        else:
            min_value = float('inf')
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
                    elif self.can_jump(curr_board, move, direction):
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

    def terminal_test(self, board: dict[Coord, str], depth) -> bool:
        """
        Check if the game is over, i.e., if all frogs of one color are on the opposite side of the board.
        """
        #if has explored the maximum depth
        if depth == 0:
            return True
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
        Apply the action to the board and return the new board state.
        assume the action are valid and is handled in previous methods
        """
        new_board = board.copy()  # Create a copy to avoid modifying the original

        # If it is a move action
        if isinstance(action, MoveAction):
            # Start coordinate of the frog
            start_coord = action.coord
            current_pos = start_coord
            # Sequence of directions to move
            directions = action.directions
            # Keep the color of the frog
            frog_color = new_board[start_coord]
            # Remove the frog from start position
            del new_board[start_coord]

            # Apply moves
            for direction in directions:
                current_pos += direction

            # Place the frog at the final position
            new_board[current_pos] = frog_color

        # If it is a grow action
        elif isinstance(action, GrowAction):
            # Determine player color character
            player_color = getattr(action, 'player_color', self.color)
            player_color_char = 'r' if player_color == PlayerColor.RED else 'b'

            # Find all frogs of the player's color
            frogs = [coord for coord, state in new_board.items() if state == player_color_char]

            # For each frog add lily pads adjacent to it
            for frog in frogs:
                for dir_r in range(-1, 2):
                    for dir_c in range(-1, 2):
                        # Skip if at frog's position
                        if dir_r == 0 and dir_c == 0:
                            continue

                        # Check if the new coordinate would be within bounds
                        new_r = frog.r + dir_r
                        new_c = frog.c + dir_c

                        # Only create a Coord if it's within the board boundaries
                        if 0 <= new_r < 8 and 0 <= new_c < 8:
                            try:
                                new_coord = Coord(new_r, new_c)
                                # If the new coordinate is empty, add a lily pad
                                if new_coord not in new_board:
                                    new_board[new_coord] = 'l'
                            except ValueError:
                                # Skip if coordinate creation fails
                                pass

        return new_board

    def evaluation_function(self, board: dict[Coord, str], color: PlayerColor) -> float:
        """
        Evaluate the board state. Higher values are better for the player.
        """

        # Count the number of frogs on the opposite side of the board and priority
        red_score = sum(1 for c, s in board.items() if s == 'r' and c.r == 7)
        blue_score = sum(1 for c, s in board.items() if s == 'b' and c.r == 0)

        # Look for other frogs on the board which are not on the opposite side
        red_partial_score = sum((c.r) for c, s in board.items() if s == 'r' and c.r != 7)
        blue_partial_score = sum((7 - c.r) for c, s in board.items() if s == 'b' and c.r != 0)

        # Overall score for each player
        red_total_score = red_score * 100 + red_partial_score
        blue_total_score = blue_score * 100 + blue_partial_score

        if color == PlayerColor.RED:
            return red_total_score - blue_total_score
        elif color == PlayerColor.BLUE:
            return blue_total_score - red_total_score
        else:
            return 0.0

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
            if state == color:
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

    def get_all_possible_jumps(self, curr: Coord, initial_board: dict[Coord, str], color : PlayerColor) -> list[Action]:
        """
        Instead of returning the end coord, it should return all possible actions
        """
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
            if landing_node not in cur_board or cur_board[landing_node] != 'l': #if not lily then it cannot land
                return False
            else: return True
        return False

    def get_best_move(self) -> Action:
        """
        Find the best move using the MinMax algorithm for the current player.
        Evaluates all possible actions (grow or move) and returns the one with the best value.
        Returns an Action object (either GrowAction or MoveAction).
        """
        # Initialize tracking variables for the best move found
        best_value = float('-inf')  # Start with worst possible value
        best_move = None  # No move selected yet

        # Try a grow action first
        # Growing adds lily pads adjacent to all frogs of the player's color
        grow_action = GrowAction()
        new_board = self.apply_action(self.board.copy(), grow_action)  # Create new board state after grow
        # Evaluate this state from opponent's perspective (minimizing player)
        grow_value = self.min_max_value(new_board, self.opposite_color(self.color), self.depth - 1, False)

        # Update best move if grow action is better than current best
        if grow_value > best_value:
            best_value = grow_value
            best_move = grow_action

        # STRATEGY 2: Try all possible move actions for each frog
        for frog in self.get_frog_coords(self.board, self.color):  # Iterate through all player's frogs
            for direction in self.get_possible_directions(self.color):  # Check each valid direction
                move = frog + direction  # Calculate potential landing position

                # Case 1: Simple move to lily pad
                if self.is_valid_move(self.board, move):
                    # Create a move action with a single direction
                    move_action = MoveAction(frog, direction)
                    new_board = self.apply_action(self.board.copy(), move_action)  # Apply move to copy of board
                    # Evaluate resulting board state from opponent's perspective
                    move_value = self.min_max_value(new_board, self.opposite_color(self.color), self.depth - 1, False)

                    # Update best move if this move is better
                    if move_value > best_value:
                        best_value = move_value
                        best_move = move_action

                # Case 2: Jump sequence (more complex)
                continue
        return best_move
