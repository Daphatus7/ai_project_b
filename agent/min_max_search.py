from typing import List

from langsmith import evaluate

from referee.game import Action, Coord, PlayerColor, Direction, MoveAction, GrowAction, board, BOARD_N


def opposite_color(color: PlayerColor) -> PlayerColor | None:
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


def get_frog_coords(curr_board: dict[Coord, str], color: PlayerColor) -> list[Coord]:
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
    for coord, state in curr_board.items():
        if state == player_color:
            frogs.append(coord)
    return frogs


def get_possible_directions(color : PlayerColor)-> list[Direction]:
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


def is_valid_move(board: dict[Coord, str], move: Coord) -> bool:
    """
    Check if the move is valid
    """
    return move in board and board[move] == 'l'


def terminal_test(board: dict[Coord, str], depth) -> bool:
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

    def print_board(self, print_board: dict[Coord, str]):
        """
        Print a text representation of the current board state.
        """
        print("  " + " ".join(str(c) for c in range(BOARD_N)))
        print("  " + "-" * (BOARD_N * 2 - 1))


        for r in range(BOARD_N):
            row = f"{r}|"
            for c in range(BOARD_N):
                cell = print_board.get(Coord(r, c))
                if cell == 'r':
                    row+="R"
                elif cell == 'b':
                    row+="B"
                elif cell == 'l':
                    row+="*"
                else:
                    row+="."
                row += " "
            print(row)
    @staticmethod
    def is_on_board(column, row) -> bool:
        return 0 <= column < BOARD_N and 0 <= row < BOARD_N

    def update_board(self, action: Action, actor: PlayerColor):
        self.apply_action(self.board, action, actor)

    def apply_action(self, new_board: dict[Coord, str], action: Action, color: PlayerColor) -> dict[Coord, str]:
        """
        Applying changes to the board
        data is "NOT" duplicated here, if want to copy, do it outside
        assume it is verified
        """
        # If it is a move action
        if isinstance(action, MoveAction):
            directions = action.directions
            
            # Case tuple
            if not isinstance(directions, (list, tuple)):
                directions = [directions]
            
            # remove the original player at the location
            frog_color = new_board.pop(action.coord)
            # move the frog
            current_pos = action.coord

            for direction in directions:
                #print("moving", current_pos, direction)
                new_c = current_pos.c + direction.c
                new_r = current_pos.r + direction.r
                if self.is_on_board( new_c, new_r):
                    current_pos = Coord(new_r, new_c)
                    if current_pos in new_board and new_board[current_pos] in ['r', 'b']:
                        current_pos = current_pos + direction
            # destination
            new_board[current_pos] = frog_color
        # grow around the player
        elif isinstance(action, GrowAction):
            for coord, state in list(new_board.items()):
                # for all frogs
                compare_color = 'r' if color == PlayerColor.RED else 'b' if color == PlayerColor.BLUE else None
                if state == compare_color:
                    # all around the frog
                    for grow_tile in self.get_grow_tiles(coord):
                        # if the tile is empty
                        if grow_tile not in new_board:
                            new_board[grow_tile] = 'l'

        return new_board

    def get_grow_tiles(self, curr: Coord) -> list[Coord]:
        """
        grow in 8 directions
        """
        grow_tiles = []

        for row, column in [(-1, -1), (-1, 0), (-1, 1), ( 0, -1), ( 0, 1), ( 1, -1), ( 1, 0), ( 1, 1)]:
            new_row = curr.r + row
            new_col = curr.c + column
            if self.is_on_board(new_col, new_row):
                grow_tiles.append(Coord(new_row, new_col))
        return grow_tiles

    def evaluation_function(self, board: dict[Coord, str], color: PlayerColor) -> float:
        """
        Evaluate the board state. Higher values are better for the player.
        """

        # Count the number of frogs on the opposite side of the board and priority
        red_score = sum(1 for c, s in board.items() if s == 'r' and c.r == 7)
        blue_score = sum(1 for c, s in board.items() if s == 'b' and c.r == 0)

        # Look for other frogs on the board which are not on the opposite side
        red_partial_score = sum((c.r) for c, s in board.items() if s == 'r' and c.r != 7)
        blue_partial_score = sum((c.r) for c, s in board.items() if s == 'b' and c.r != 0)


        # Overall score for each player
        red_total_score = red_score * 100 + red_partial_score
        blue_total_score = blue_score * 100 + blue_partial_score

        if color == PlayerColor.RED:
            return red_total_score - blue_total_score
        elif color == PlayerColor.BLUE:
            return blue_total_score - red_total_score
        else:
            return 0.0

    def temp_jump_only_consider_one(self, color: PlayerColor) -> list[Direction]:
        pass

    def get_all_possible_jumps(self, start_coord: Coord, initial_board: dict[Coord, str], color : PlayerColor) -> list[Action]:
        """
        Instead of returning the end coord, it should return all possible actions
        """
        moves: List[Action] = []

        def dfs(curr: Coord, path: List[Direction], visited: set[Coord]) -> None:
            can_jump = False
            for direction in get_possible_directions(color):
                converted_coord = self.convert_direction_to_coord(direction)
                n_r = curr.r + converted_coord[0]
                n_c = curr.c + converted_coord[1]
                if self.is_on_board(n_c, n_r): # check if the next cell is on the board
                    neighbour = Coord(n_r, n_c)
                    if not neighbour in visited: # check if the next cell is already visited
                        if self.can_jump(initial_board, neighbour, direction):
                            landing_node = Coord(neighbour.r + converted_coord[0], neighbour.c + converted_coord[1])

                            can_jump = True
                            dfs(landing_node, path + [direction], visited | {neighbour, landing_node})
            #cannot jump anymore
            if not can_jump:
                if path:
                    moves.append(MoveAction(start_coord, tuple(path)))

        dfs(start_coord, [], {start_coord})
        return moves

    def convert_direction_to_coord(self, direction: Direction) -> tuple[int, int]:
        """
        Convert the direction to a coordinate
        """
        if direction == Direction.Up:
            return -1, 0
        elif direction == Direction.Down:
            return 1, 0
        elif direction == Direction.Left:
            return 0, -1
        elif direction == Direction.Right:
            return 0, 1
        elif direction == Direction.UpLeft:
            return -1, -1
        elif direction == Direction.UpRight:
            return -1, 1
        elif direction == Direction.DownLeft:
            return 1, -1
        elif direction == Direction.DownRight:
            return 1, 1

    def can_jump(self, cur_board : dict[Coord, str], neighbour_node, direction) -> bool:
        """
        Check if the frog can jumpc z
        1. the neighbour_node node must be in the board
        2. the neighbour_node must be a frog
        """
        if neighbour_node not in cur_board:#
            return False

        neighbour = cur_board[neighbour_node]
        if neighbour not in ['r', 'b']: # is a frog
            return False
        #check the direction
        direction_coord = self.convert_direction_to_coord(direction)
        landing_c = neighbour_node.c + direction_coord[1]
        landing_r = neighbour_node.r + direction_coord[0]

        if not self.is_on_board(landing_c, landing_r):
            return False
        landing_node = Coord(landing_r, landing_c)
        # Landing position must be a lily pad and not occupied by another frog
        return landing_node in cur_board and cur_board[landing_node] == 'l'

    def evaluate_min_max(self, curr_board: dict[Coord, str], color : PlayerColor, depth: int, maximizing_player : bool):
        value = float('-inf') if maximizing_player else float('inf')
        # for each frog on the board
        grow_value = self.min_max_value(self.apply_action(curr_board.copy(), GrowAction(), color), color, depth, not maximizing_player)
        value = max(value, grow_value) if maximizing_player else min(value, grow_value)
        for frog in get_frog_coords(curr_board, color):
            # for each possible direction
            for direction in get_possible_directions(color):  # possible moves should also include jumps
                move_r = frog.r + direction.r
                move_c = frog.c + direction.c
                if not self.is_on_board(move_c, move_r):
                    continue
                move = frog + direction
                # if is a lily pad -> end
                if is_valid_move(curr_board, move):
                    # apply the move (also removes the lily pad)
                    move_action = MoveAction(frog, direction)
                    move_value = self.min_max_value(self.apply_action(curr_board.copy(), move_action, color), opposite_color(color), depth, not maximizing_player)
                    value = max(value, move_value) if maximizing_player else min(value, move_value)

                # if the next location is a frog -> apply the move ->
                elif self.can_jump(curr_board, move, direction):
                    # need to check every possible jumps
                    for jump in self.get_all_possible_jumps(move, curr_board, color):
                        jump_value = self.min_max_value(self.apply_action(curr_board.copy(), jump, color), opposite_color(color), depth, not maximizing_player)
                        value = max(value, jump_value) if maximizing_player else min(value, jump_value)
        return value

    def min_max_value(self, curr_board: dict[Coord, str], color : PlayerColor, depth: int, maximizing_player : bool) -> float | List[Action] | Action:
        new_depth = depth - 1
        if terminal_test(curr_board, new_depth):
            return self.evaluation_function(curr_board, color)
        if maximizing_player:
            return self.evaluate_min_max(curr_board, color, new_depth, False)
        else:
            return self.evaluate_min_max(curr_board, color, new_depth, True)

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
        new_board = self.apply_action(self.board.copy(), grow_action, self.color)  # Create new board state after grow
        # Evaluate this state from opponent's perspective (minimizing player)
        grow_value = self.min_max_value(new_board, opposite_color(self.color), explore_depth, True)

        # Update best move if grow action is better than current best
        if grow_value > max_value:
            max_value = grow_value
            best_move = grow_action

        # for each frog, we evaluate each possible move.
        for frog_location in get_frog_coords(self.board, self.color):
            for direction in get_possible_directions(self.color):
                move_r = frog_location.r + direction.r
                move_c = frog_location.c + direction.c
                if not self.is_on_board(move_c, move_r):
                    continue
                move = frog_location + direction
                # if the next cell is a lilypad
                if is_valid_move(self.board, move):
                    # Create a move action with a single direction
                    move_action = MoveAction(frog_location, direction)
                    value = self.min_max_value( self.apply_action(self.board.copy(), move_action, self.color),
                                                     opposite_color(self.color),
                                                     explore_depth, True)
                    if value > max_value:
                        max_value = value
                        best_move = move_action

                elif self.can_jump(self.board, move, direction):
                    # need to check every possible jumps
                    for jump in self.get_all_possible_jumps(move, self.board, self.color):
                        value = self.min_max_value(self.apply_action(self.board.copy(), jump, self.color),
                                                   opposite_color(self.color),
                                                   explore_depth, True)
                        if value > max_value:
                            max_value = value
                            best_move = jump
        print("best move", best_move)
        return best_move
