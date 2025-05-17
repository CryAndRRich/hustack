## Description
We are given a rectangular maze represented by an $N \times M$ binary matrix. Each cell in the matrix is either:
- $0$: an empty cell where movement is allowed.
- $1$: a wall that blocks movement.

From a given starting cell (provided as 1-indexed coordinates), you can move in four directions: up, down, left, or right—if the adjacent cell is empty. The goal is to determine the minimum number of steps required to exit the maze. A cell is considered an exit if it is located on the boundary of the maze (i.e., in the first or last row, or in the first or last column). Note that if the starting cell is on the boundary, exiting the maze may require just one step.

**Input:**
- **Line 1**: Four integers: $N$, $M$, $r$, $c$  
  - $N$ and $M$ are the numbers of rows and columns (with $1 \leq N, M \leq 999$).
  - $(r, c)$ is the starting cell (given in 1-indexed coordinates).
- **Next $N$ lines**: Each line contains M integers (either $0$ or $1$) representing the maze layout.

**Output:**
- A single integer indicating the minimum number of steps required to exit the maze. If there is no valid path, output $-1$.

## Approach 1: Breadth-First Search (BFS)
BFS explores the maze in layers, guaranteeing that the first time an exit cell is reached, the path taken is the shortest possible. We use a queue to maintain the cells to be processed along with their current distance from the starting cell.

### Time Complexity:
- In the worst-case scenario, each cell is processed once, leading to a time complexity of $O(N.M)$.

### Space Complexity:
- The queue and the visited matrix both require $O(N.M)$ space.

## Approach 2: Depth-First Search (DFS) + Backtracking (Failed)
DFS explores a path as far as possible before backtracking to try alternative paths. To find the shortest exit path, DFS is implemented with backtracking: when an exit cell (i.e., a cell on the boundary) is reached, the number of steps is recorded and compared with a global minimum. The algorithm then continues exploring, backtracking when necessary to ensure that all potential paths are examined.

### Time Complexity:
- In the worst-case, DFS may explore many paths and could have an exponential time complexity. However, early backtracking (cutting off paths that already exceed the best found solution) often improves the average-case performance.

### Space Complexity:
- The recursion stack and the visited matrix both use up to $O(N.M)$ space in the worst-case scenario.
