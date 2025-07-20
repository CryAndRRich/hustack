## Description
A rectangle of size $n \times m$ is divided into $1 \times 1$ square cells, each colored either black or white. The rectangle is represented by an $n \times m$ matrix $A$, where $A(i,j)=1$ means that the cell in row $i$ and column $j$ is black, and $A(i,j)=0$ means that the cell is white (the rows and columns are numbered starting from $1$). A subrectangle of the board is defined as a rectangle whose top-left cell is $(1, 1)$. Determine the subrectangle of the given board that contains only black cells and has the maximum area.

**Input:**
- **Line $1$:** Two positive integers $n$ and $m$ $(1 \leq n,m \leq 50)$
- **Lines $2$ to $n+1$:** Each line contains the elements of a row of matrix $A$

**Output:** Print the area of the largest subrectangle found

## Approach: Prefix Sum
For each cell $(i, j)$ ($1$-indexed), compute:
$$S[i][j] = A[i][j] + S[i-1][j] + S[i][j-1] - S[i-1][j-1]$$
This makes $S[i][j]$ the sum of the submatrix from $(1, 1)$ to $(i, j)$. Iterate over all possible bottom-right corners $(i, j)$ of subrectangles starting at $(1,1)$. For a subrectangle with area $i \times j$, if $S[i][j] == i * j$ then all cells are black

**Time Complexity:** $O(n \times m)$

**Space Complexity:** $O(n \times m)$
