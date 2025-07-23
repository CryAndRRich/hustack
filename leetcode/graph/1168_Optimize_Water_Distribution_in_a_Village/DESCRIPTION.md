## Description
There are $n$ houses in a village. We want to supply water for all the houses by building wells and laying pipes.

For each house $i$, we can either build a well inside it directly with cost $wells[i - 1]$ (note the $-1$ due to $0$-indexing), or pipe in water from another well to it. The costs to lay pipes between houses are given by the array $pipes$ where each $pipes[j] = [house1_j, house2_j, cost_j]$ represents the cost to connect $house1_j$ and $house2_j$ together using a pipe. Connections are bidirectional, and there could be multiple valid connections between the same two houses with different costs.

Return the minimum total cost to supply water to all houses.

## Example
### Example 1:
**Input**: $n = 3, wells = [1,2,2], pipes = [[1,2,1],[2,3,1]]$  
**Output**: $3$  
**Explanation**: The best strategy is to build a well in the first house with cost $1$ and connect the other houses to it with cost $2$ so the total cost is $3$

### Example 2:
**Input**: $n = 2, wells = [1,1], pipes = [[1,2,1],[1,2,2]]$  
**Output**: $2$  
**Explanation**: We can supply water with cost two using one of the three options:
- Option 1: The total cost will be $2$
    - Build a well inside house $1$ with cost $1$
    - Build a well inside house $2$ with cost $1$
- Option 2: The total cost will be $2$
    - Build a well inside house $1$ with cost $1$
    - Connect house 2 with house $1$ with cost $1$
- Option 3: The total cost will be $2$
    - Build a well inside house $2$ with cost $1$
    - Connect house 1 with house $2$ with cost $1$

## Constraints:
- $2 \leq n \leq 10^4$
- $wells.length == n$
- $0 \leq wells[i] \leq 10^5$
- $1 \leq pipes.length \leq 10^4$
- $pipes[j].length == 3$
- $1 \leq house1_j, house2_j \leq n$
- $0 \leq cost_j \leq 10^5$
- $house1_j \neq house2_j$