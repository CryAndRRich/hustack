## Description
In an infinite chess board with coordinates from $-infinity$ to $+infinity$, you have a knight at square $[0, 0]$.

A knight has $8$ possible moves it can make. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.


Return the minimum number of steps needed to move the knight to the square [x, y]. It is guaranteed the answer exists.

## Example
### Example 1:
**Input**: $x = 2, y = 1$  
**Output**: $1$  
**Explanation**: $[0, 0] \rightarrow [2, 1]$

### Example 2:
**Input**: $x = 5, y = 5$  
**Output**: $4$  
**Explanation**: $[0, 0] \rightarrow [2, 1] \rightarrow [4, 2] \rightarrow [3, 4] \rightarrow [5, 5]$
 
## Constraints:
- $-300 \leq x, y \leq 300$
- $0 \leq |x| + |y| \leq 300$