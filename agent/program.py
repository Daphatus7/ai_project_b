# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from referee.game.board import CellState, BOARD_N
from referee.game import Action, Coord, PlayerColor, Direction, MoveAction, GrowAction
import heapq

from typing import List


"""
Code from our project A - a* pathfinding
"""

# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part A: Single Player Freckers
# Node class to represent each state in the A* search
class Node:
    def __init__(self, coord, g, h):
        self.coord = coord  # Coordinates of the node
        self.g = g  # Set to be the path length
        self.h = h  # Heuristic cost to the end node
        self.f = g + h  # Total cost (g + h)

    def __eq__(self, other):
        return self.coord == other.coord

    def __lt__(self, other):
        return self.f < other.f

# Manhattan distance heuristic - minimum distance to any end position
def h_cost(start: Coord, ends: list[Coord]) -> int:
    if not ends:  # if no valid ends, return a large value
        return BOARD_N * 2  # max possible distance on the board
    return min(abs(start.r - end.r) + abs(start.c - end.c) for end in ends)

def pathfinding( curr_board: dict[Coord, str], start: Coord, my_color: PlayerColor) -> int | None:
    """
    Simplified a* that only track the cost of the path
    """
    action_list = []
    closed = set()

    goal_row = BOARD_N - 1 if my_color == PlayerColor.RED else 0
    goals =[]
    for column in range(BOARD_N):
        if valid_landing_spot(curr_board, Coord(goal_row, column)):
            goals.append(Coord(goal_row, column))

    if not goals:
        return None
    elif start in goals:
        return BOARD_N - 1 #maximum reward because it is already on the goal row

    start_node = Node(start, 0, h_cost(start, goals))
    min_g ={start: 0}
    heapq.heappush(action_list, start_node)

    while action_list:
        current = heapq.heappop(action_list)
        closed.add(current.coord)

        # g is set to be the path legnth
        if current.coord in goals:
            return current.g

        # all neighbours
        for direction in get_possible_directions(my_color):
            r_vector = current.coord.r + direction.r
            c_vector = current.coord.c + direction.c
            if not is_on_board(r_vector, c_vector):
                continue
            new_coord = Coord(r_vector, c_vector)
            if new_coord in closed:
                continue
            if can_jump(curr_board, new_coord, direction):
                land = new_coord + direction
                new_g = current.g + 1
                coord2 = land
            elif valid_landing_spot(curr_board, new_coord):
                new_g = current.g + 1
                coord2 = new_coord
            else:
                continue
            if coord2 in min_g and new_g >= min_g[coord2]:
                continue
            min_g[coord2] = new_g
            heapq.heappush(action_list, Node(coord2, new_g, h_cost(coord2, goals)))
    return None


# Check if the coordinates are within the board boundaries
def is_on_board(r, c):
    return 0 <= r < BOARD_N and 0 <= c < BOARD_N

# Check if red frog can jump over a blue frog in a given direction
def can_jump(curr_board: dict[Coord, str], neighbour_node: Coord, direction: Direction) -> bool:
    """
    Check if the frog can jump
    1. the neighbour_node must be in the board
    2. the neighbour_node must be a frog
    3. there must be a valid landing spot
    """
    if neighbour_node not in curr_board:
        return False

    neighbour = curr_board[neighbour_node]
    if neighbour not in ['r', 'b']:  # is a frog
        return False

    # check the direction
    landing_c = neighbour_node.c + direction.c
    landing_r = neighbour_node.r + direction.r

    if not is_on_board(landing_c, landing_r):
        return False
    landing_node = Coord(landing_r, landing_c)
    # Landing position must be a lily pad and not occupied by another frog
    return landing_node in curr_board and curr_board[landing_node] == 'l'

# Check if position is a valid landing spot
def valid_landing_spot(board, coord):
    if coord not in board:
        return False
    return (board[coord] == 'l' and
            board[coord] != 'r' and
            board[coord] != 'b')





# A* search algorithm


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
        self._search_depth = 4 # error when search depth is 1
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
        #print("referee update actions",  action)
        #print("----------Before-------------")
        #self.__print_board()
        self.__brain.update_board(action, color)
        #printout board
        #print("----------After-------------")
        #self.__print_board()



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
    2.all frogs reached goal row
    """
    # if has explored the maximum depth
    if depth == 0:
        return True
        # Check if all RED frogs are at row 7
    red_frogs = [coord for coord, state in board.items() if state == 'r']
    if all(frog.r == BOARD_N - 1 for frog in red_frogs) and red_frogs:
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


    WEIGHTS = [1, 2, 4, 8, 16, 32, 64, 128, 258]

    def evaluation_function(self, curr_board: dict[Coord, str], my_player_color: PlayerColor) -> float:
        """
        Evaluate the board state. Higher values are better for the player.
        """
        frog_color = None
        opponent_color = None
        if my_player_color == PlayerColor.RED:
            frog_color = 'r'
            opponent_color = 'b'
        elif my_player_color == PlayerColor.BLUE:
            frog_color = 'b'
            opponent_color = 'r'


        def get_all_frogs(color: str) -> list[Coord]:
            frogs = []
            for cell, state in list(curr_board.items()):
                if state == color:
                    frogs.append(cell)
            return frogs

        def evaluate_distance_score(frogs: list[Coord], color: PlayerColor)-> float:
            score = 0
            goal_row = BOARD_N - 1 if color == PlayerColor.RED else 0
            for frog in frogs:
                # estimate the cost to a valid row column
                distance = abs(goal_row - frog.r)
                if distance <= 4: # use pathfinding to estimate a valid row column
                    a_star_distance = pathfinding(curr_board, frog, color)
                    if a_star_distance is not None:
                        frog_score = self.WEIGHTS[len(self.WEIGHTS) - a_star_distance - 1]
                    else:
                        frog_score = self.WEIGHTS[len(self.WEIGHTS) - distance - 1]
                else:
                # estimate the closest distance to the goal row
                    frog_score = self.WEIGHTS[len(self.WEIGHTS) - distance - 1]
                score += frog_score
            return score

        my_score = evaluate_distance_score(get_all_frogs(frog_color), my_player_color)
        opponent_score = evaluate_distance_score(get_all_frogs(opponent_color), opposite_color(my_player_color))
        #print ("my versus opponent score", my_score)
        total_score = my_score - opponent_score

        return total_score

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
        Check if the frog can jump
        1. the neighbour_node node must be in the board
        2. the neighbour_node must be a frog
        """
        return can_jump(cur_board, neighbour_node, direction)

    def evaluate_min_max(self, curr_board: dict[Coord, str], color: PlayerColor, depth: int,
                         alpha, beta,
                         maximizing_player: bool):
        value = float('-inf') if maximizing_player else float('inf')
        # for each frog on the board
        grow_value = self.min_max_value(self.apply_action(curr_board.copy(), GrowAction(), color), color, depth,
                                        alpha, beta,
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
                    result = self.min_max_value(self.apply_action(curr_board.copy(), move_action, color),
                                                    opposite_color(color), depth,
                                                    alpha, beta,
                                                    not maximizing_player)

                    if maximizing_player:
                        value = max(value, result)
                        alpha = max(alpha, result)
                        if beta <= alpha:
                            break

                    else:
                        value = min(value, result)
                        beta = min(beta, result)
                        if beta <= alpha:
                            break

                # if the next location is a frog -> apply the move ->
                elif self.can_jump(curr_board, move, direction):
                    # need to check every possible jumps
                    for jump in self.get_all_possible_jumps(frog, curr_board, color):
                        result = self.min_max_value(self.apply_action(curr_board.copy(), jump, color),
                                                        opposite_color(color), depth,
                                                        alpha,
                                                        beta,
                                                        not maximizing_player)
                        if maximizing_player:
                            value = max(value, result)
                            alpha = max(alpha, result)
                            if beta <= alpha:
                                break
                        else:
                            value = min(value, result)
                            beta = min(beta, result)
                            if beta <= alpha:
                                break
            if beta <= alpha:
                break
        return value
    def min_max_value(self, curr_board: dict[Coord, str], color: PlayerColor, depth: int,
                      alpha, beta,
                      maximizing_player: bool) -> float | List[Action] | Action:
        new_depth = depth - 1
        if terminal_test(curr_board, new_depth):
            return self.evaluation_function(curr_board, self.color)
        return self.evaluate_min_max(curr_board, color, new_depth, alpha, beta, maximizing_player)

    def get_best_move(self, my_color: PlayerColor) -> Action:
        """
        Finding the best move
        1. because we only care about the next move
        2. we estimate the impact of next move
        3. then we conclude if the next move is the best
        """
        #goal_row = 0 if my_color == PlayerColor.BLUE else BOARD_N - 1


        #1) instant win, simple steps and jumps
        # for frog in get_frog_coords(self.board, my_color):
        #     for direction in get_possible_directions(my_color):
        #         # compute candidate step
        #         new_r = frog.r + direction.r
        #         new_c = frog.c + direction.c
        #
        #         if not self.is_on_board(new_c, new_r):
        #             continue
        #         # straight move to the goal row
        #         if is_valid_move(self.board, Coord(new_r, new_c)):
        #             if new_r == goal_row:
        #                 return MoveAction(frog, direction)
        #
        #         # check jump possibility
        #         jump_start = Coord(new_r, new_c)
        #
        #         if self.can_jump(self.board, jump_start, direction):
        #             for jump in self.get_all_possible_jumps(frog, self.board, my_color):
        #                 landing_r, landing_c = jump.coord.r, jump.coord.c
        #                 for d in jump.directions:
        #                     landing_r += d.r
        #                     landing_c += d.c
        #                     if not self.is_on_board(landing_c, landing_r):
        #                         continue
        #                     if landing_r == goal_row:
        #                         return jump

        # min value
        max_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        best_move = None  # No move selected yet
        explore_depth = self.depth - 1

        # Try a grow action first
        # Growing adds lily pads adjacent to all frogs of the player's color
        grow_action = GrowAction()
        new_board = self.apply_action(self.board.copy(), grow_action, self.color)  # Create new board state after grow
        # Evaluate this state from opponent's perspective (minimizing player)
        grow_value = self.min_max_value(new_board, opposite_color(self.color), explore_depth,
                                        alpha, beta,
                                        False)

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
                # if the next cell is a lilypad
                if is_valid_move(self.board, Coord(move_r, move_c)):
                    # Create a move action with a single direction
                    move_action = MoveAction(frog_location, direction)
                    value = self.min_max_value(
                        self.apply_action(self.board.copy(), move_action, my_color),
                               opposite_color(my_color),
                               explore_depth,
                               alpha,
                               beta,
                               False)
                    if value > max_value:
                        max_value = value
                        best_move = move_action
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break

                jump_start = Coord(move_r, move_c)
                if self.can_jump(self.board, jump_start, direction):
                    # need to check every possible jumps
                    for jump in self.get_all_possible_jumps(frog_location, self.board, my_color):
                        value = self.min_max_value(self.apply_action(self.board.copy(), jump, my_color),
                                                   opposite_color(my_color),
                                                   explore_depth,
                                                   alpha,
                                                   beta,
                                                   False)
                        if value > max_value:
                            max_value = value
                            best_move = jump
                        alpha = max(alpha, value)
                        if beta <= alpha:
                            break
            if beta <= alpha:
                break
        print("best move", best_move)
        return best_move


