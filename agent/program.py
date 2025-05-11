# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

import heapq
from typing import List
from referee.game.board import BOARD_N, Board
from referee.game import Action, Coord, PlayerColor, Direction, MoveAction, GrowAction
import random


def get_grow_tiles(curr: Coord) -> list[Coord]:
    """
    grow in 8 directions
    """
    grow_tiles = []

    for row, column in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        new_row = curr.r + row
        new_col = curr.c + column
        if is_on_board(new_col, new_row):
            grow_tiles.append(Coord(new_row, new_col))
    return grow_tiles

class MyBoard:

    def __init__(self, board: list[list[str]], red_frogs_positions = None, blue_frogs_positions = None):
        self.board = board
        self.red_frogs_positions = set()
        self.blue_frogs_positions = set()

        if red_frogs_positions is not None and blue_frogs_positions is not None:
            self.red_frogs_positions = set(red_frogs_positions) # this is a copied instance
            self.blue_frogs_positions = set(blue_frogs_positions) # this is a copied instance
        else:
            for row in range(BOARD_N):
                for column in range(BOARD_N):
                    cell = board[row][column]
                    if cell == 'r':
                        self.red_frogs_positions.add(Coord(row, column))
                    elif cell == 'b':
                        self.blue_frogs_positions.add(Coord(row, column))

    #how to update frog positions
    def get_frog_coords(self, color: PlayerColor) -> list[Coord]:
        return list(self.red_frogs_positions if color == PlayerColor.RED else self.blue_frogs_positions)

    def apply_action(self, action: Action, color: PlayerColor) -> list[list[str]]:
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
            frog_color = self.board[action.coord.r][action.coord.c]
            self.board[action.coord.r][action.coord.c] = '.'  # remove the frog

            # move the frog
            curr_r = action.coord.r
            curr_c = action.coord.c
            # for each direction - for jumps
            for direction in directions:
                new_r = curr_r + direction.r
                new_c = curr_c + direction.c
                if is_on_board(new_r, new_c):
                    # considering jump
                    if self.board[curr_r][curr_c] in ['r', 'b']:
                        curr_r += direction.r
                        curr_c += direction.c
            # destination
            self.board[curr_r][curr_c] = frog_color
            self.__update_frog_positions(action.coord, Coord(curr_r, curr_c))
        # grow around the player
        elif isinstance(action, GrowAction):
            frogs = self.get_frog_coords(color)
            for frog in frogs:
                # for all frogs
                for grow_tile in get_grow_tiles(frog):
                    # if the tile is empty
                    if grow_tile not in self.board:
                        self.board[grow_tile.r][grow_tile.c] = 'l'
        return self.board

    def is_valid_move(self, move: Coord) -> bool:
        """
        Check if the move is valid
        """
        return self.board[move.r][move.c] == 'l'

    def __update_frog_positions(self, frog: Coord, frog_new_coord: Coord):
        """
        the frog must be either red or blue
        """
        if frog in self.red_frogs_positions:
            self.red_frogs_positions.remove(frog)
            self.red_frogs_positions.add(frog_new_coord)
        elif frog in self.blue_frogs_positions:
            self.blue_frogs_positions.remove(frog)
            self.blue_frogs_positions.add(frog_new_coord)

    def valid_landing_spot(self, coord : Coord) -> bool:
        return (self.board[coord.r][coord.c] == 'l' and
                self.board[coord.r][coord.c] != 'r' and
                self.board[coord.r][coord.c] != 'b')

    def print_board(self):
        """
        Print a text representation of the current board state.
        """
        print("  " + " ".join(str(c) for c in range(BOARD_N)))
        print("  " + "-" * (BOARD_N * 2 - 1))

        for r in range(BOARD_N):
            row = f"{r}|"
            for c in range(BOARD_N):
                cell = self.board[r][c]
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

    def can_jump(self, neighbour_node: Coord, direction: Direction) -> bool:
        """
        Check if the frog can jump
        1. the neighbour_node must be in the board
        2. the neighbour_node must be a frog
        3. there must be a valid landing spot
        """
        neighbour = self.board[neighbour_node.r][neighbour_node.c]
        if neighbour not in ['r', 'b']:  # is a frog
            return False

        # check the direction
        landing_c = neighbour_node.c + direction.c
        landing_r = neighbour_node.r + direction.r
        if is_on_board(landing_r, landing_c):
            # Landing position must be a lily pad and not occupied by another frog
            return self.board[landing_r][landing_c] == 'l'
        return False

    def deep_copy(self)-> "MyBoard":
        """
        Create a deep copy of the board
        """
        return MyBoard(self.board.copy(), self.red_frogs_positions, self.blue_frogs_positions)

class BoardState:
    def __init__(self,board :list[list[str]], action: Action, evaluation : float, alpha : float, beta : float, red_frog_positions, blue_frogs_positions):
        self.board : list[list[str]] = board.copy()
        self.red_frogs_positions = red_frog_positions.copy()
        self.blue_frogs_positions = blue_frogs_positions.copy()
        self.action = action
        self.evaluation = evaluation
        self.alpha = alpha
        self.beta = beta

##https://www.youtube.com/watch?v=QYNRvMolN20&t=201s
##https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-5-zobrist-hashing/
class TranspositionTable:
    def __init__(self):
        self.size = 1000000
        self.cache = {}
        random.seed(39)
        self.zobrist_table = self.__generate_zobrist_table()

    def __generate_zobrist_table(self) -> list[list[list[int]]]:
        """
        Generate the Zobrist by assigning a unique random number to each cell representing 3 different states
        """
        return [[[random.getrandbits(64) for _ in range(3)] for _ in range(BOARD_N)] for _ in range(BOARD_N)]

    def get_hashing_key(self, board: list[list[str]]) -> int:
        """
        Generate the zobrist key for the given board
        """
        hash_key = 0
        for row in range(BOARD_N):
            for column in range(BOARD_N):
                if board is not None:
                    cell = board[row][column]
                    if cell is not None:
                        cell_value = self.__get_cell_value(cell)
                        if cell_value is not None:
                            #xor each cell value
                            hash_key ^= self.zobrist_table[row][column][cell_value]
        return hash_key

    def __get_cell_value(self, cell: str) -> int | None:
        """
        Get the cell value
        lily pad = 0
        red frog = 1
        blue frog = 2
        """
        if cell == 'l':
            return 0
        elif cell == 'r':
            return 1
        elif cell == 'b':
            return 2
        else:
            return None

    def store_board_state(self, board: list[list[str]], board_state: BoardState):
        """
        Store the evaluation
        """
        hash_key = self.get_hashing_key(board)

        if hash_key in self.cache:
            self.cache[hash_key] = board_state
        else:
            if len(self.cache) >= self.size:
                # remove the oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[hash_key] = board_state

    def get_cached_board_state(self, curr_board : MyBoard) -> BoardState | None:
        """
        get the cached board
        1. get the hash key
        2. check if the hash key is in the cache
        """
        return self.cache.get(self.get_hashing_key(curr_board.board))

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

def pathfinding(my_board : MyBoard, start: Coord, my_color: PlayerColor) -> int | None:
    """
    Simplified a* that only track the cost of the path
    """
    action_list = []
    closed = set()
    curr_board = my_board.board
    goal_row = BOARD_N - 1 if my_color == PlayerColor.RED else 0
    goals =[] # all possible goal positions
    for column in range(BOARD_N):
        if my_board.valid_landing_spot(Coord(goal_row, column)):
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
            if my_board.can_jump(new_coord, direction):
                land = new_coord + direction
                new_g = current.g + 1
                coord2 = land
            elif my_board.valid_landing_spot(new_coord):
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

# A* search algorithm
class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self.__color = color
        self._board: list[list[str]] = [['.' for _ in range(BOARD_N)] for _ in range(BOARD_N)]
        """ # Setup initial board state
        print("\nSetting up initial board state...") """

        # Corner lily pads
        for row in [0, BOARD_N - 1]:
            for column in [0, BOARD_N - 1]:
                self._board[row][column] = 'l'

        # Middle row lily pads
        for row in [1, BOARD_N - 2]:
            for column in range(1, BOARD_N - 1):
                self._board[row][column] = 'l'

        # Initial RED and BLUE pieces
        for c in range(1, BOARD_N - 1):
            self._board[0][column] = 'r'
            self._board[BOARD_N - 1][column] = 'b'

        # Set minimax search depth
        self._search_depth = 5 # error when search depth is 1
        self.__brain = MinMaxSearch(self._board, self._search_depth, self.__color)





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
        # Print remaining referee time
        print(f"Referee time remaining: {referee['time_remaining']} seconds")
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

def terminal_test(board: MyBoard, depth) -> bool:
    """
    check if satisfier terminal condition
    1. has reached maximum depth
    2.all frogs reached goal row
    """
    # if has explored the maximum depth
    if depth == 0:
        return True
        # Check if all RED frogs are at row 7
    red_frogs = board.red_frogs_positions
    if all(frog.r == BOARD_N - 1 for frog in red_frogs) and red_frogs:
        return True
    # Check if all BLUE frogs are at row 0
    blue_frogs = board.blue_frogs_positions
    if all(frog.r == 0 for frog in blue_frogs) and blue_frogs:
        return True

    return False


def convert_direction_to_coord(direction: Direction) -> tuple[int, int]:
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


class MinMaxSearch:
    """
    MinMaxSearch class for implementing the MinMax algorithm for game AI.
    """

    def __init__(self, board: list[list[str]], depth, color: PlayerColor):
        """
        The initial state of the board
        """
        self.board = MyBoard(board)
        self.depth = depth
        self.color = color
        self.best_move = None
        self.cache = TranspositionTable()

    def update_board(self, action: Action, color: PlayerColor):
        self.board.apply_action(action, color)

    WEIGHTS = [1, 2, 4, 8, 16, 32, 64, 200, 850] # 62


    def evaluation_function(self, curr_board: MyBoard,
                            alpha: float,
                            beta: float,
                            current_move: Action,
                            my_player_color: PlayerColor) -> float:
        """
        Evaluate the board state. Higher values are better for the player.
        """


        def evaluate_distance_score(frogs: list[Coord], color: PlayerColor)-> float:
            score = 0
            goal_row = BOARD_N - 1 if color == PlayerColor.RED else 0
            for frog in frogs:
                # estimate the cost to a valid row column
                distance = abs(goal_row - frog.r)
                if 4 >= distance >= 1: # use pathfinding to estimate a valid row column
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

        # Get Frogs
        my_frogs = curr_board.get_frog_coords(my_player_color)
        opponent_frogs = curr_board.get_frog_coords(opposite_color(my_player_color))
        # Calculate the distance score for both players
        my_score = evaluate_distance_score(my_frogs, my_player_color)
        opponent_score = evaluate_distance_score(opponent_frogs, opposite_color(my_player_color))
        total_score = my_score - opponent_score

        # store the evaluation result in the cache

        board_state = BoardState(curr_board.board, current_move, total_score, alpha, beta,
                                 curr_board.red_frogs_positions,
                                 curr_board.blue_frogs_positions)

        self.cache.store_board_state(curr_board.board, board_state)
        return total_score

    def get_all_possible_jumps(self, start_coord: Coord, initial_board: MyBoard, color: PlayerColor) -> list[Action]:
        """
        Instead of returning the end coord, it should return all possible actions
        """
        moves: List[Action] = []

        def dfs(curr: Coord, path: List[Direction], visited: set[Coord]) -> None:
            can_jump = False
            for direction in get_possible_directions(color):
                converted_coord = convert_direction_to_coord(direction)
                n_r = curr.r + converted_coord[0]
                n_c = curr.c + converted_coord[1]
                if is_on_board(n_c, n_r):  # check if the next cell is on the board
                    neighbour = Coord(n_r, n_c)
                    if not neighbour in visited:  # check if the next cell is already visited
                        if initial_board.can_jump(neighbour, direction):
                            landing_node = Coord(neighbour.r + converted_coord[0], neighbour.c + converted_coord[1])
                            can_jump = True
                            dfs(landing_node, path + [direction], visited | {neighbour, landing_node})
            # cannot jump anymore
            if not can_jump:
                if path:
                    moves.append(MoveAction(start_coord, tuple(path)))
        dfs(start_coord, [], {start_coord})
        return moves

    def evaluate_min_max(self, curr_board: MyBoard, color: PlayerColor, depth: int,
                         alpha, beta,
                         current_move : Action,
                         maximizing_player: bool):
        value = float('-inf') if maximizing_player else float('inf')
        # for each frog on the board

        cached_board_state = self.cache.get_cached_board_state(curr_board)
        if cached_board_state is not None: # that means, we have already calculated the part of the board,
            # recreate the board from the database
            cached_board = MyBoard(
                cached_board_state.board,
                cached_board_state.red_frogs_positions,
                cached_board_state.blue_frogs_positions)
            # if the board is already evaluated -> continue from here
            # if the board is not evaluated -> evaluate it again
            print("cached board state", cached_board_state.evaluation)
            return self.min_max_value(cached_board,
                                      color,
                                      depth,
                                      cached_board_state.alpha,
                                      cached_board_state.beta,
                                      cached_board_state.action,
                                      maximizing_player)

        print("-----Depth-----", depth)
        #1.  copy the board
        new_board = curr_board.deep_copy()
        # apply the action
        new_board.apply_action(GrowAction(), color)
        grow_value = self.min_max_value(new_board, color, depth,
                                        alpha, beta,
                                        GrowAction(),
                                        not maximizing_player)
        value = max(value, grow_value) if maximizing_player else min(value, grow_value)

        for frog in curr_board.get_frog_coords(color):
            # for each possible direction
            for direction in get_possible_directions(color):  # possible moves should also include jumps
                move_r = frog.r + direction.r
                move_c = frog.c + direction.c
                if not is_on_board(move_c, move_r):
                    continue
                move = frog + direction
                # if is a lily pad -> end
                if curr_board.is_valid_move(move):
                    # apply the move (also removes the lily pad)
                    move_action = MoveAction(frog, direction)
                    # copy board
                    curr_board_copy = curr_board.deep_copy()
                    # apply the action
                    curr_board_copy.apply_action(move_action, color)
                    # next level
                    result = self.min_max_value(curr_board_copy,
                                                    opposite_color(color), depth,
                                                    alpha, beta,
                                                    move_action,
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
                elif curr_board.can_jump(move, direction):
                    # need to check every possible jumps
                    for jump in self.get_all_possible_jumps(frog, curr_board, color):
                        #copy the board
                        curr_board_copy = curr_board.deep_copy()
                        # apply the action
                        curr_board_copy.apply_action(jump, color)
                        # evaluate
                        result = self.min_max_value(curr_board_copy,
                                                        opposite_color(color), depth,
                                                        alpha,
                                                        beta,
                                                        jump,
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
    def min_max_value(self, curr_board: MyBoard, color: PlayerColor, depth: int,
                      alpha, beta,
                      current_move: Action,
                      maximizing_player: bool) -> float | List[Action] | Action:
        new_depth = depth - 1
        if terminal_test(curr_board, new_depth):
            #if the board is already evaluated -> continue from here
            cached_board_state = self.cache.get_cached_board_state(curr_board)
            if cached_board_state is not None:
                return cached_board_state.evaluation
            #if not cached, we need to evaluate it again
            evaluation = self.evaluation_function(curr_board,
                                                  alpha,
                                                  beta,
                                                  current_move,
                                                  self.color)
            # store the evaluation
            board_state = BoardState(curr_board.board, current_move, evaluation, alpha, beta,
                                     curr_board.red_frogs_positions,
                                     curr_board.blue_frogs_positions)
            self.cache.store_board_state(curr_board.board, board_state)
            return evaluation
        return self.evaluate_min_max(curr_board, color, new_depth,
                                     alpha, beta,
                                     current_move,
                                     maximizing_player)
    def get_best_move(self, my_color: PlayerColor) -> Action:
        """
        Finding the best move
        1. because we only care about the next move
        2. we estimate the impact of next move
        3. then we conclude if the next move is the best
        """

        # min value
        max_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        best_move = None  # No move selected yet
        explore_depth = self.depth - 1

        cached_board_state = self.cache.get_cached_board_state(self.board)
        if cached_board_state is not None: # that means, we have already calculated the part of the board,
            return cached_board_state.action
        # Try a grow action first
        # Growing adds lily pads adjacent to all frogs of the player's color
        grow_action = GrowAction()
        # copy
        new_board = self.board.deep_copy()
        # apply
        new_board.apply_action(grow_action, self.color)
        # Evaluate this state from opponent's perspective (minimizing player)
        grow_value = self.min_max_value(new_board, opposite_color(self.color), explore_depth,
                                        alpha, beta,
                                        grow_action,
                                        False)
        print("jump", grow_value)
        # Update best move if grow action is better than current best
        if grow_value > max_value:
            max_value = grow_value
            best_move = grow_action

        # for each frog, we evaluate each possible move.
        for frog_location in self.board.get_frog_coords(my_color):
            for direction in get_possible_directions(my_color):
                move_r = frog_location.r + direction.r
                move_c = frog_location.c + direction.c
                if not is_on_board(move_c, move_r):
                    continue
                # if the next cell is a lilypad
                if self.board.is_valid_move(Coord(move_r, move_c)):
                    move_action = MoveAction(frog_location, direction)
                    # copy
                    curr_board_copy = self.board.deep_copy()
                    # apply
                    curr_board_copy.apply_action(move_action, my_color)
                    # Create a move action with a single direction
                    value = self.min_max_value(
                               curr_board_copy,
                               opposite_color(my_color),
                               explore_depth,
                               alpha,
                               beta,
                               move_action,
                               False)
                    print("move", value)
                    if value > max_value:
                        max_value = value
                        best_move = move_action
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break

                jump_start = Coord(move_r, move_c)
                if self.board.can_jump(jump_start, direction):
                    # need to check every possible jumps
                    for jump in self.get_all_possible_jumps(frog_location, self.board, my_color):
                        # copy
                        curr_board_copy = self.board.deep_copy()
                        # apply
                        curr_board_copy.apply_action(jump, my_color)
                        value = self.min_max_value(curr_board_copy,
                                                   opposite_color(my_color),
                                                   explore_depth,
                                                   alpha,
                                                   beta,
                                                   jump,
                                                   False)
                        print("jump", value)
                        if value > max_value:
                            max_value = value
                            best_move = jump
                        alpha = max(alpha, value)
                        if beta <= alpha:
                            break
            if beta <= alpha:
                break
        return best_move
