from typing import List

from langsmith import evaluate

from referee.game import Action, Coord, PlayerColor, Direction, MoveAction, GrowAction, board


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


    def evaluate_min_max(self, curr_board: dict[Coord, str], color : PlayerColor, depth: int, maximizing_player : bool):
        depth -= 1
        value = float('-inf') if maximizing_player else float('inf')
        # for each frog on the board
        grow_value = self.min_max_value(self.apply_action(curr_board, GrowAction), color, depth, not maximizing_player)
        value = max(value, grow_value) if maximizing_player else min(value, grow_value)
        for frog in self.get_frog_coords(curr_board, color):
            # for each possible direction
            for direction in self.get_possible_directions(color):  # possible moves should also include jumps
                move = frog + direction
                # if is a lily pad -> end
                if self.is_valid_move(curr_board, move):
                    # apply the move (also removes the lily pad)
                    move_value = self.min_max_value(self.apply_action(curr_board, move), self.opposite_color(color), depth, not maximizing_player)
                    value = max(value, move_value) if maximizing_player else min(value, move_value)

                # if the next location is a frog -> apply the move ->
                elif self.can_jump(curr_board, move, direction):
                    # need to check every possible jumps
                    for jump in self.get_all_possible_jumps(move, curr_board, color):
                        jump_value = self.min_max_value(self.apply_action(curr_board, jump), self.opposite_color(color), depth, not maximizing_player)
                        value = max(value, jump_value) if maximizing_player else min(value, jump_value)
        return value

    def min_max_value(self, curr_board: dict[Coord, str], color : PlayerColor, depth: int, maximizing_player : bool) -> float | List[Action] | Action:
        new_depth = depth - 1
        if self.terminal_test(curr_board, new_depth):
            return self.evaluation_function(curr_board, color)
        if maximizing_player:
            return self.evaluate_min_max(curr_board, color, new_depth, not maximizing_player)
        else:
            return self.evaluate_min_max(curr_board, color, new_depth, not maximizing_player)

    def is_valid_move(self, board: dict[Coord, str], move: Coord) -> bool:
        """
        Check if the move is valid
        """
        return move in board and board[move] == 'l'

    def terminal_test(self, board: dict[Coord, str], depth) -> bool:
        """
        check if satisfier terminal condition
        1. has reached maximum depth
        2.
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
                current_pos = current_pos + direction

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
        #get all frogs on the board
        player_color = ''
        if color == PlayerColor.RED:
            player_color = 'r'
        elif color == PlayerColor.BLUE:
            player_color = 'b'
        for coord, state in board.items():
            if state == player_color:
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

    def get_all_possible_jumps(self, start_coord: Coord, initial_board: dict[Coord, str], color : PlayerColor) -> list[Action]:
        """
        Instead of returning the end coord, it should return all possible actions
        """
        moves: List[Action] = []

        def dfs(curr: Coord, path: List[Direction], visited: set[Coord]) -> None:
            can_jump = False
            for direction in self.get_possible_directions(color):
                neighbour = curr + direction
                if not neighbour in visited:
                    if self.can_jump(initial_board, neighbour, direction):
                        landing_node = curr + direction * 2
                        can_jump = True
                        dfs(landing_node, path + [direction], visited | {neighbour})
            if not can_jump and path:
                moves.append(Action(start_coord, path))

        dfs(start_coord, [], {start_coord})
        print("all possible jumps", moves)
        return moves
        # while stack:
        #     exploring_node = stack.pop()
        #     if exploring_node in reachable_nodes:
        #         continue
        #     reachable_nodes.add(exploring_node)
        #     #explore all the neighbors
        #     for direction in self.get_possible_directions(color):
        #         #check if the next node can jump
        #         node_can_jump = exploring_node + direction #a frog
        #         if self.can_jump(initial_board, node_can_jump, direction): # if the neighbour is a frog
        #             #move current location to the next node
        #             landing_node = exploring_node + direction * 2
        #             stack.append(landing_node)
        #         else:# where the jump sequence is ended - so no more
        #             ...
        # return moves


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
        Finding the best move
        1. because we only care about the next move
        2. we estimate the impact of next move
        3. then we conclude if the next move is the best
        """

        #min value
        max_value = float('-inf')
        best_move = None  # No move selected yet
        explore_depth = self.depth - 1


        # Try a grow action first
        # Growing adds lily pads adjacent to all frogs of the player's color
        grow_action = GrowAction()
        new_board = self.apply_action(self.board.copy(), grow_action)  # Create new board state after grow
        # Evaluate this state from opponent's perspective (minimizing player)
        grow_value = self.min_max_value(new_board, self.opposite_color(self.color), explore_depth, False)

        # Update best move if grow action is better than current best
        if grow_value > max_value:
            max_value = grow_value
            best_move = grow_action

        # for each frog, we evaluate each possible move.
        for frog_location in self.get_frog_coords(self.board, self.color):
            print("checking the frog", frog_location)
            for direction in self.get_possible_directions(self.color):
                move = frog_location + direction
                # if the next cell is a lilypad
                if self.is_valid_move(self.board, move):
                    # Create a move action with a single direction
                    move_action = MoveAction(frog_location, direction)
                    value = self.min_max_value( self.apply_action(self.board, move_action),
                                                     self.opposite_color(self.color),
                                                     explore_depth, False)
                    if value > max_value:
                        max_value = max_value
                        best_move = move_action

                elif self.can_jump(self.board, move, direction):
                        # need to check every possible jumps
                        for jump in self.get_all_possible_jumps(move, self.board, self.color):
                            move_action = MoveAction(frog_location, jump)
                            value = self.min_max_value(self.apply_action(self.board, move_action),
                                                       self.opposite_color(self.color),
                                                       explore_depth, False)
                            if value > max_value:
                                max_value = value
                                best_move = move_action
        print("best move", best_move)
        return best_move
