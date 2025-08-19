## Description
You are given a $2D$ array grid of size $m \times n$, and an integer $k$. There are $k$ cells in grid containing the values from $1$ to $k$ exactly once, and the rest of the cells have a value $0$.

You can start at any cell, and move from a cell to its neighbors (up, down, left, or right). You must find a path in grid which:
- Visits each cell in grid exactly once.
- Visits the cells with values from $1$ to $k$ in order.

Return a $2D$ array result of size $(m \times n) \times 2$, where $result[i] = [x_i, y_i]$ represents the $i^{th}$ cell visited in the path. If there are multiple such paths, you may return any one.

If no such path exists, return an empty array.

## Example
### Example 1:
**Input**: $grid = [[0,0,0],[0,1,2]], k = 2$  
**Output**: $[[0,0],[1,0],[1,1],[1,2],[0,2],[0,1]]$  

### Example 2:
**Input**: $grid = [[1,0,4],[3,0,2]], k = 4$  
**Output**: $[]$  

## Constraints:
- $1 \leq m == grid.length \leq 5$
- $1 \leq n == grid[i].length \leq 5$
- $1 \leq k \leq m \times n$
- $0 \leq grid[i][j] \leq k$
- grid contains all integers between $1$ and $k$ exactly once.