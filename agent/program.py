# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
from agent.min_max_search import MinMaxSearch
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
        next_move = self.__brain.get_best_move()

        # Check if the next move is a valid action
        return next_move


    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """
        if action is None: return

        print("referee update actions",  action)
        print("----------Before-------------")
        self.__print_board()
        self.__brain = MinMaxSearch(self._board, self._search_depth, self.__color)
        self.__brain.update_board(action, color)
        #printout board
        print("----------After-------------")
        self.__print_board()
        # There are two possible action types: MOVE and GROW. Below we check
        # which type of action was played and print out the details of the
        # action for demonstration purposes. You should replace this with your
        # own logic to update your agent's internal game state representation.
        match action:
            case MoveAction(coord, dirs):
                dirs_text = ", ".join([str(dir) for dir in dirs])
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Directions: {dirs_text}")
            case GrowAction():
                print(f"Testing: {color} played GROW action")
            case _:
                raise ValueError(f"Unknown action type: {action}")
