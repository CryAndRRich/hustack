## Description
A rectangle of size $n \times m$ is divided into  unit-square $1 \times 1$ with two colors black and white. The rectangle is represented by a matrix $A(n \times m)$, where $A(i,j)=1$ means that the unit-square at row $i$ and column $j$ is black, and $A(i,j)=0$ means that the unit-square is white. Find the sub-rectangle of the given rectangle containing only black unit-squares such that the area is maximal

**Input:**
- **Line $1$:** Two positive integers $n$ and $m$ $(1 \leq n,m \leq 1000)$
- **Lines $2$ to $n+1$:** Each line contains the elements of a row of matrix $A$

**Output:** Write the area of the sub-rectangle found

## Approach: Stack
- **Build a histogram** for each row:  
    - Maintain an array $h[1..m]$ where $h[j]$ counts consecutive $1$’s up to the current row in column $j$.  
    - If the cell is $1$, increment $h[j]$; otherwise reset $h[j]$ to $0$

- **Largest Rectangle in Histogram** problem: For each updated $h[]$, use a stack to compute the maximal rectangle area:
    - Append a sentinel $0$ height
    - Iterate through indices; whenever the current height is less than the stack’s top height:
        - Pop the stack, compute $area = popped\_height × width$ (distance to new stack top or start)
    - Update max area
    - Push current index onto the stack

- Repeat for all n rows, tracking the global maximum

**Time Complexity:** $O(nm)$
- Building histograms: $O(nm)$  
- Histogram max-rectangle per row: $O(m)$  

**Space Complexity:** $O(m)$
