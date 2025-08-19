## Description
The diameter of a tree is the number of edges in the longest path in that tree.

There is an undirected tree of $n$ nodes labeled from $0$ to $n - 1$. You are given a $2D$ array $edges$ where $edges.length == n - 1$ and $edges[i] = [a_i, b_i]$ indicates that there is an undirected edge between nodes $a_i$ and $b_i$ in the tree.

Return the diameter of the tree.

## Example
### Example 1:
**Input**: $edges = [[0,1],[0,2]]$  
**Output**: $2$  
**Explanation**: The longest path of the tree is the path $1 - 0 - 2$.

### Example 2:
**Input**: $edges = [[0,1],[1,2],[2,3],[1,4],[4,5]]$  
**Output**: $4$  
**Explanation**: The longest path of the tree is the path $3 - 2 - 1 - 4 - 5$.
 
## Constraints:
- $n == edges.length + 1$
- $1 \leq n \leq 10^4$
- $0 \leq a_i, b_i < n$
- $a_i \neq b_i$