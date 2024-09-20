"""
CS311 Programming Assignment 1: Search

Full Name: Frank Bautista

Brief description of my heuristic:

TODO Briefly describe your heuristic and why it is more efficient
"""

import argparse, itertools, random, sys
from typing import Callable, List, Optional, Sequence, Tuple


# You are welcome to add constants, but do not modify the pre-existing constants

# Problem size 
BOARD_SIZE = 3
TOP_RANGE = list(range(1, BOARD_SIZE - 1))
BOTTOM_RANGE = list(range(BOARD_SIZE**2 - BOARD_SIZE, BOARD_SIZE**2 - 1))
LEFT_RANGE = list(range(BOARD_SIZE, BOARD_SIZE**2 - BOARD_SIZE, BOARD_SIZE))
RIGHT_RANGE = list(range(2*BOARD_SIZE - 1, BOARD_SIZE**2 - 1, BOARD_SIZE))

# The goal is a "blank" (0) in bottom right corner
GOAL = tuple(range(1, BOARD_SIZE**2)) + (0,)


def inversions(board: Sequence[int]) -> int:
    """Return the number of times a larger 'piece' precedes a 'smaller' piece in board"""
    return sum(
        (a > b and a != 0 and b != 0) for (a, b) in itertools.combinations(board, 2)
    )


class Node:
    def __init__(self, state: Sequence[int], parent: "Node" = None, cost=0):
        """Create Node to track particular state and associated parent and cost

        State is tracked as a "row-wise" sequence, i.e., the board (with _ as the blank)
        1 2 3
        4 5 6
        7 8 _
        is represented as (1, 2, 3, 4, 5, 6, 7, 8, 0) with the blank represented with a 0

        Args:
            state (Sequence[int]): State for this node, typically a list, e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8]
            parent (Node, optional): Parent node, None indicates the root node. Defaults to None.
            cost (int, optional): Cost in moves to reach this node. Defaults to 0.
        """
        self.state = tuple(state)  # To facilitate "hashable" make state immutable
        self.parent = parent
        self.cost = cost

    def is_goal(self) -> bool:
        """Return True if Node has goal state"""
        return self.state == GOAL

    def expand(self) -> List["Node"]:
        """
        Expand current node into possible child nodes with corresponding parent and cost

        Thoughts:
        There 2-4 possible movies for the blank space, depending on its location:
            1. If the blank is in the middle of the board, it can move in any direction
            2. If the blank is on the edge of the board, it can move in 3 directions
            3. If the blank is in the corner of the board, it can move in 2 directions
        
        We know there are only 4 corners in any sized board, so we can check the following positions:
            1. Top left corner: position [0]
            2. Top right corner: position [BOARD_SIZE - 1]
            3. Bottom left corner: position [(BOARD_SIZE - 1) * BOARD_SIZE] or [BOARD_SIZE**2 - BOARD_SIZE]
            4. Bottom right corner: position [BOARD_SIZE**2 - 1]
        
        We can then check the following positions to see if the blank is on the edge of the board:
            1. Top edge: positions [1, 2, ..., BOARD_SIZE - 2] 
            2. Bottom edge: positions [BOARD_SIZE**2 - BOARD_SIZE + 1, ..., BOARD_SIZE**2 - 2]
            3. Left edge: positions [BOARD_SIZE, 2*BOARD_SIZE, ..., BOARD_SIZE**2 - BOARD_SIZE]
            4. Right edge: positions [2*BOARD_SIZE - 1, 3*BOARD_SIZE - 1, ..., BOARD_SIZE**2 - 1]
        
        If the blank is not in the corner or edge, then it is in the middle of the board and can move in any direction (else statement)

        """
        
        index_of_blank = self.state.index(0)
        is_corner, what_corner = is_corner_blank(index_of_blank)
        is_edge, what_edge = is_edge_blank(index_of_blank)
        row = index_of_blank // BOARD_SIZE
        col = index_of_blank % BOARD_SIZE


        if is_corner:
            return self.corner_blank_states(what_corner)
        elif is_edge:
            return self.edge_blank_states(what_edge, row, col)
        else:
            return self.middle_blank_states(row, col)
        

    def _swap(self, row1: int, col1: int, row2: int, col2: int) -> Sequence[int]:
        """Swap values in current state between row1,col1 and row2,col2, returning new "state" to construct a Node"""
        state = list(self.state)
        state[row1 * BOARD_SIZE + col1], state[row2 * BOARD_SIZE + col2] = (
            state[row2 * BOARD_SIZE + col2],
            state[row1 * BOARD_SIZE + col1],
        )
        return state

    def corner_blank_states(self, position_of_blank: str ) -> List["Node"]:
        """
        Move the blank space in a corner node.
        If the blank space is in the corner, return the possible moves the blank space can make.
        
        Args: 
            what_corner (str): The corner the blank space is in

        Returns:
            List["Node"]: List of possible moves the blank space can make
        """
        children = []
        if position_of_blank == "top-left":
            children = [
                Node(self._swap(0, 0, 0, 1), self, self.cost + 1), # Move blank to the right
                Node(self._swap(0, 0, 1, 0), self, self.cost + 1) # Move blank down
            ]
        elif position_of_blank == "top-right":
            children = [
                Node(self._swap(0, BOARD_SIZE - 1, 0, BOARD_SIZE - 2), self, self.cost + 1), # Move blank to the left
                Node(self._swap(0, BOARD_SIZE - 1, 1, BOARD_SIZE - 1), self, self.cost + 1) # Move blank down
            ]
        elif position_of_blank == "bottom-left":
            children = [
                Node(self._swap(BOARD_SIZE - 1, 0, BOARD_SIZE - 1, 1), self, self.cost + 1), # Move blank to the right
                Node(self._swap(BOARD_SIZE - 1, 0, BOARD_SIZE - 2, 0), self, self.cost + 1) # Move blank up
            ]
        elif position_of_blank == "bottom-right":
            children = [
                Node(self._swap(BOARD_SIZE - 1, BOARD_SIZE - 1, BOARD_SIZE - 1, BOARD_SIZE - 2), self, self.cost + 1), # Move blank to the left
                Node(self._swap(BOARD_SIZE - 1, BOARD_SIZE - 1, BOARD_SIZE - 2, BOARD_SIZE - 1), self, self.cost + 1) # Move blank up
            ]
        return children
    
    def edge_blank_states(self, position_of_blank: str, row: int, col: int) -> List["Node"]:
        """
        Move the blank space in an edge node.
        If the blank space is in the edge, return the possible moves the blank space can make.
        
        Args: 
            what_edge (str): The edge the blank space is in
            row (int): The row the blank space is in
            col (int): The column the blank space is in

        Returns:
            List["Node"]: List of possible moves the blank space can make
        """

        if position_of_blank == "top":
            return [
                Node(self._swap(row, col, row, col - 1), self, self.cost + 1) if col > 0 else None, # Move blank to the left
                Node(self._swap(row, col, row, col + 1), self, self.cost + 1) if col < BOARD_SIZE - 1 else None, # Move blank to the right
                Node(self._swap(row, col, row + 1, col), self, self.cost + 1) # Move blank down
            ]
        elif position_of_blank == "bottom":
            return [
                Node(self._swap(row, col, row, col - 1), self, self.cost + 1) if col > 0 else None, # Move blank to the left
                Node(self._swap(row, col, row, col + 1), self, self.cost + 1) if col < BOARD_SIZE - 1 else None, # Move blank to the right
                Node(self._swap(row, col, row - 1, col), self, self.cost + 1) # Move blank up
            ]
        elif position_of_blank == "left":
            return [
                Node(self._swap(row, col, row - 1, col), self, self.cost + 1) if row > 0 else None, # Move blank up
                Node(self._swap(row, col, row + 1, col), self, self.cost + 1) if row < BOARD_SIZE - 1 else None, # Move blank down
                Node(self._swap(row, col, row, col + 1), self, self.cost + 1) # Move blank to the right
            ]
        elif position_of_blank == "right":
            return [
                Node(self._swap(row, col, row - 1, col), self, self.cost + 1) if row > 0 else None, # Move blank up
                Node(self._swap(row, col, row + 1, col), self, self.cost + 1) if row < BOARD_SIZE - 1 else None, # Move blank down
                Node(self._swap(row, col, row, col - 1), self, self.cost + 1) # Move blank to the left
            ]
    
    def middle_blank_states(self, row: int, col: int) -> List["Node"]:
        """
        Move the blank space in a middle node.
        If the blank space is in the middle, return the possible moves the blank space can make.
        
        Args: 
            row (int): The row the blank space is in
            col (int): The column the blank space is in

        Returns:
            List["Node"]: List of possible moves the blank space can make
        """

        return [
            Node(self._swap(row, col, row - 1, col), self, self.cost + 1) if row > 0 else None, # Move blank up
            Node(self._swap(row, col, row + 1, col), self, self.cost + 1) if row < BOARD_SIZE - 1 else None, # Move blank down
            Node(self._swap(row, col, row, col - 1), self, self.cost + 1) if col > 0 else None, # Move blank to the left
            Node(self._swap(row, col, row, col + 1), self, self.cost + 1) if col < BOARD_SIZE - 1 else None # Move blank to the right
        ]


    def __str__(self):
        return str(self.state)

    # The following methods enable Node to be used in types that use hashing (sets, dictionaries) or perform comparisons. Note
    # that the comparisons are performed exclusively on the state and ignore parent and cost values.

    def __hash__(self):
        return self.state.__hash__()

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


def bfs(initial_board: Sequence[int], max_depth=12) -> Tuple[Optional[Node], int]:
    """Perform breadth-first search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """

    # TODO: Implement BFS. Your function should return a tuple containing the solution node and number of unique node explored
    return None, 0

def is_edge_blank(blank_index: int) -> Tuple[bool, str]:
    """
    Check if blank is on the edge of the board.
    If so return True and either "top", "bottom", "left", or "right" depending on the edge.
    
    Args: 
        blank_index (int): Index of the blank space in the board

    Returns:
        Tuple[bool, str]: Tuple of boolean value and string
    """
    
    if blank_index in TOP_RANGE:
        return True, "top"
    elif blank_index in BOTTOM_RANGE:
        return True, "bottom"
    elif blank_index in LEFT_RANGE:
        return True, "left"
    elif blank_index in RIGHT_RANGE:
        return True, "right"
    else:
        return False, ""

def is_corner_blank(blank_index: int) -> Tuple[bool, str]:
    """
    Check if blank is on the corner of the board.
    If so return True and either "top-left", "top-right", "bottom-left", or "bottom-right" depending on the corner.
    
    Args: 
        blank_index (int): Index of the blank space in the board

    Returns:
        Tuple[bool, str]: Tuple of boolean value and string
    """
    

    if blank_index == 0:
        return True, "top-left"
    elif blank_index == BOARD_SIZE - 1:
        return True, "top-right"
    elif blank_index == BOARD_SIZE**2 - BOARD_SIZE:
        return True, "bottom-left"
    elif blank_index == BOARD_SIZE**2 - 1:
        return True, "bottom-right"
    else:
        return False, ""





def manhattan_distance(node: Node) -> int:
    """Compute manhattan distance f(node), i.e., g(node) + h(node)"""
    # TODO: Implement the Manhattan distance heuristic (sum of Manhattan distances to goal location)



def custom_heuristic(node: Node) -> int:
    # TODO: Implement and document your _admissable_ heuristic function
    return 0


def astar(
    initial_board: Sequence[int],
    max_depth=12,
    heuristic: Callable[[Node], int] = manhattan_distance,
) -> Tuple[Optional[Node], int]:
    """Perform astar search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 12.
        heuristic (_Callable[[Node], int], optional): Heuristic function. Defaults to manhattan_distance.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    # TODO: Implement A* search. Make sure that your code uses the heuristic function provided as
    # an argument so that the test code can switch in your custom heuristic (i.e., do not "hard code"
    # manhattan distance as the heuristic)
    return None, sys.maxsize

if __name__ == "__main__":

    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(
        description="Run search algorithms in random inputs"
    )
    parser.add_argument(
        "-a",
        "--algo",
        default="bfs",
        help="Algorithm (one of bfs, astar, astar_custom)",
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=1000,
        help="Number of iterations",
    )
    parser.add_argument(
        "-s",
        "--state",
        type=str,
        default=None,
        help="Execute a single iteration using this board configuration specified as a string, e.g., 123456780",
    )

    args = parser.parse_args()

    num_solutions = 0
    num_cost = 0
    num_nodes = 0

    if args.algo == "bfs":
        algo = bfs
    elif args.algo == "astar":
        algo = astar
    elif args.algo == "astar_custom":
        algo = lambda board: astar(board, heuristic=custom_heuristic)
    else:
        raise ValueError("Unknown algorithm type")

    if args.state is None:
        iterations = args.iter
        while iterations > 0:
            init_state = list(range(BOARD_SIZE**2))
            random.shuffle(init_state)

            # A problem is only solvable if the parity of the initial state matches that
            # of the goal.
            if inversions(init_state) % 2 != inversions(GOAL) % 2:
                continue

            solution, nodes = algo(init_state)
            if solution:
                num_solutions += 1
                num_cost += solution.cost
                num_nodes += nodes

            iterations -= 1
    else:
        # Attempt single input state
        solution, nodes = algo([int(s) for s in args.state])
        if solution:
            num_solutions = 1
            num_cost = solution.cost
            num_nodes = nodes

    if num_solutions:
        print(
            "Iterations:",
            args.iter,
            "Solutions:",
            num_solutions,
            "Average moves:",
            num_cost / num_solutions,
            "Average nodes:",
            num_nodes / num_solutions,
        )
    else:
        print("Iterations:", args.iter, "Solutions: 0")
