## Description
You have a graph of $n$ nodes. You are given an integer $n$ and an array edges where $edges[i] = [a_i, b_i]$ indicates that there is an edge between $a_i$ and $b_i$ in the graph.

Return the number of connected components in the graph.

## Example
### Example 1:
**Input**: $n = 5, edges = [[0,1],[1,2],[3,4]]$  
**Output**: $2$

### Example 2:
**Input**: $n = 5, edges = [[0,1],[1,2],[2,3],[3,4]]$  
**Output**: $1$
 
## Constraints:
- $1 <= n <= 2000$
- $1 <= edges.length <= 5000$
- $edges[i].length == 2$
- $0 <= a_i <= b_i < n$
- $a_i != b_i$
- There are no repeated edges