# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.board import CellState, BOARD_N


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def _convert_board(self, board_data) -> dict[Coord, str]:
        converted = {}
        for coord, cell in board_data.items():
            # Extract state from CellState if needed
            if isinstance(cell, CellState):
                state = cell.state
            else:
                state = cell

            # Map to internal representation
            if state == PlayerColor.RED:
                converted[coord] = 'r'
            elif state == PlayerColor.BLUE:
                converted[coord] = 'b'
            elif str(state) == "LilyPad":  # Handle enum string representation
                converted[coord] = 'l'
            else:
                converted[coord] = '.'  # Empty cells
        return converted

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        """ # Print a clear separator for initialization debugging
        print("\n" + "="*50)
        print("INITIALIZING AGENT")
        print("="*50)
        
        # Debug referee dictionary content
        print(f"Referee params: {referee}")
        
        # Store our color
        """
        self.__color = color
        print(f"Agent color: {self.__color}")

        # Initialize with a dictionary-based board representation
        self._board : dict[Coord, chr] = {}

        """ # Setup initial board state
        print("\nSetting up initial board state...") """

        # # Empty cells - don't need empty cells
        # for r in range(BOARD_N):
        #     for c in range(BOARD_N):
        #         self._board[Coord(r, c)] = CellState()

        # Corner lily pads
        for r in [0, BOARD_N - 1]:
            for c in [0, BOARD_N - 1]:
                self._board[Coord(r, c)] = 'l'

        # Middle row lily pads
        for r in [1, BOARD_N - 2]:
            for c in range(1, BOARD_N - 1):
                self._board[Coord(r, c)] = 'l'

        # Initial RED and BLUE pieces
        for c in range(1, BOARD_N - 1):
            self._board[Coord(0, c)] = 'r'
            self._board[Coord(BOARD_N - 1, c)] = 'b'

        # Set minimax search depth
        self._search_depth = 3 # error when search depth is 1
        self.__brain = MinMaxSearch(self._board, self._search_depth, self.__color)

        #self.__print_board()

    def __print_board(self):
        """
        Print a text representation of the current board state.
        """
        print("  " + " ".join(str(c) for c in range(BOARD_N)))
        print("  " + "-" * (BOARD_N * 2 - 1))


        for r in range(BOARD_N):
            row = f"{r}|"
            for c in range(BOARD_N):
                cell = self._board.get(Coord(r, c))
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


    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        # Below we have hardcoded two actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.
        next_move = self.__brain.get_best_move(self.__color)

        # Check if the next move is a valid action
        return next_move


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """
        print("referee update actions",  action)
        print("----------Before-------------")
        self.__print_board()
        self.__brain.update_board(action, color)
        #printout board
        print("----------After-------------")
        self.__print_board()
        # # There are two possible action types: MOVE and GROW. Below we check
        # # which type of action was played and print out the details of the
        # # action for demonstration purposes. You should replace this with your
        # # own logic to update your agent's internal game state representation.
        # match action:
        #     case MoveAction(coord, dirs):
        #         dirs_text = ", ".join([str(dir) for dir in dirs])
        #         print(f"Testing: {color} played MOVE action:")
        #         print(f"  Coord: {coord}")
        #         print(f"  Directions: {dirs_text}")
        #     case GrowAction():
        #         print(f"Testing: {color} played GROW action")
        #     case _:
        #         raise ValueError(f"Unknown action type: {action}")


from typing import List

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
    # get all frogs on the board
    player_color = ''
    if color == PlayerColor.RED:
        player_color = 'r'
    elif color == PlayerColor.BLUE:
        player_color = 'b'
    for coord, state in curr_board.items():
        if state == player_color:
            frogs.append(coord)
    return frogs


def get_possible_directions(color: PlayerColor) -> list[Direction]:
    """
    Get the possible moves for the player
    """
    # if red ->
    match color:
        case PlayerColor.RED:
            return [Direction.Right, Direction.Left, Direction.Down, Direction.DownLeft, Direction.DownRight]
        # if blue ->
        case PlayerColor.BLUE:
            return [Direction.Right, Direction.Left, Direction.Up, Direction.UpLeft, Direction.UpRight]
        # if none ->
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
    # if has explored the maximum depth
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
                    row += "R"
                elif cell == 'b':
                    row += "B"
                elif cell == 'l':
                    row += "*"
                else:
                    row += "."
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
                # print("moving", current_pos, direction)
                new_c = current_pos.c + direction.c
                new_r = current_pos.r + direction.r
                if self.is_on_board(new_c, new_r):
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

        for row, column in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            new_row = curr.r + row
            new_col = curr.c + column
            if self.is_on_board(new_col, new_row):
                grow_tiles.append(Coord(new_row, new_col))
        return grow_tiles

    WEIGHTS = [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        100
    ]

    def evaluation_function(self, curr_board: dict[Coord, str], my_color: PlayerColor) -> float:
        """
        Evaluate the board state. Higher values are better for the player.
        """
        """
            # 1 distance : closer to the goal the higher points - where the enemy 
            # 2 movements : c
            # 3 grow count
            # opponent
                # 1.
        """


        frog_color_char = None
        opponent_color_char = None
        if my_color == PlayerColor.RED:
            frog_color_char = 'r'
            opponent_color_char = 'b'
        elif my_color == PlayerColor.BLUE:
            frog_color_char = 'b'
            opponent_color_char = 'r'
        else:
            assert False
        goal_row = 0 if my_color == PlayerColor.RED else BOARD_N - 1

        def get_all_frog(color_char : str) -> list[Coord]:
            frogs = []
            for cell, state in list(curr_board.items()):
                if state == color_char:
                    frogs.append(cell)
            return frogs

        def evaluate_distance_score(frogs : list[Coord]) -> float:
            score = 0
            for frog in frogs:
                distance = abs(goal_row - frog.r)
                print(distance)
                score += self.WEIGHTS[distance]
            return score

        def evaluate_opponent_distance_score(frogs : list[Coord]) -> float:
            score = 0
            for frog in frogs:
                distance = abs(goal_row - frog.r)
                score += self.WEIGHTS[distance]
            return score

        def evaluate_space_score(frog_color : PlayerColor) -> float:
            """
            The more freedom the player has the higher the score
            """
            score = 0
            return score

        def evaluate_opponent_space_score(frogs : list[Coord]) -> float:
            score = 0
            return score

        total_score = evaluate_distance_score(get_all_frog(
            frog_color_char,
        )) - evaluate_opponent_distance_score(
            get_all_frog(
                opponent_color_char,
            )
        )
        return total_score


    def temp_jump_only_consider_one(self, color: PlayerColor) -> list[Direction]:
        pass

    def get_all_possible_jumps(self, start_coord: Coord, initial_board: dict[Coord, str], color: PlayerColor) -> list[
        Action]:
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
                if self.is_on_board(n_c, n_r):  # check if the next cell is on the board
                    neighbour = Coord(n_r, n_c)
                    if not neighbour in visited:  # check if the next cell is already visited
                        if self.can_jump(initial_board, neighbour, direction):
                            landing_node = Coord(neighbour.r + converted_coord[0], neighbour.c + converted_coord[1])

                            can_jump = True
                            dfs(landing_node, path + [direction], visited | {neighbour, landing_node})
            # cannot jump anymore
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

    def can_jump(self, cur_board: dict[Coord, str], neighbour_node, direction) -> bool:
        """
        Check if the frog can jumpc z
        1. the neighbour_node node must be in the board
        2. the neighbour_node must be a frog
        """
        if neighbour_node not in cur_board:  #
            return False

        neighbour = cur_board[neighbour_node]
        if neighbour not in ['r', 'b']:  # is a frog
            return False
        # check the direction
        direction_coord = self.convert_direction_to_coord(direction)
        landing_c = neighbour_node.c + direction_coord[1]
        landing_r = neighbour_node.r + direction_coord[0]

        if not self.is_on_board(landing_c, landing_r):
            return False
        landing_node = Coord(landing_r, landing_c)
        # Landing position must be a lily pad and not occupied by another frog
        return landing_node in cur_board and cur_board[landing_node] == 'l'

    def evaluate_min_max(self, curr_board: dict[Coord, str], color: PlayerColor, depth: int, maximizing_player: bool):
        value = float('-inf') if maximizing_player else float('inf')
        # for each frog on the board
        grow_value = self.min_max_value(self.apply_action(curr_board.copy(), GrowAction(), color), color, depth,
                                        not maximizing_player)
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
                    move_value = self.min_max_value(self.apply_action(curr_board.copy(), move_action, color),
                                                    opposite_color(color), depth, not maximizing_player)
                    value = max(value, move_value) if maximizing_player else min(value, move_value)

                # if the next location is a frog -> apply the move ->
                elif self.can_jump(curr_board, move, direction):
                    # need to check every possible jumps
                    for jump in self.get_all_possible_jumps(move, curr_board, color):
                        jump_value = self.min_max_value(self.apply_action(curr_board.copy(), jump, color),
                                                        opposite_color(color), depth, not maximizing_player)
                        value = max(value, jump_value) if maximizing_player else min(value, jump_value)
        return value

    def min_max_value(self, curr_board: dict[Coord, str], color: PlayerColor, depth: int,
                      maximizing_player: bool) -> float | List[Action] | Action:
        new_depth = depth - 1
        if terminal_test(curr_board, new_depth):
            return self.evaluation_function(curr_board, color)
        if maximizing_player:
            return self.evaluate_min_max(curr_board, color, new_depth, False)
        else:
            return self.evaluate_min_max(curr_board, color, new_depth, True)

    def get_best_move(self, my_color: PlayerColor) -> Action:
        """
        Finding the best move
        1. because we only care about the next move
        2. we estimate the impact of next move
        3. then we conclude if the next move is the best
        """

        # min value
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
        for frog_location in get_frog_coords(self.board, my_color):
            for direction in get_possible_directions(my_color):
                move_r = frog_location.r + direction.r
                move_c = frog_location.c + direction.c
                if not self.is_on_board(move_c, move_r):
                    continue
                move = frog_location + direction
                # if the next cell is a lilypad
                if is_valid_move(self.board, move):
                    # Create a move action with a single direction
                    move_action = MoveAction(frog_location, direction)
                    value = self.min_max_value(self.apply_action(self.board.copy(), move_action, self.color),
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
