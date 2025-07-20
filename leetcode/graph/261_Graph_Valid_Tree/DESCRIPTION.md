## Description
You have a graph of $n$ nodes labeled from $0$ to $n - 1$. You are given an integer $n$ and a list of edges where $edges[i] = [a_i, b_i]$ indicates that there is an undirected edge between nodes $a_i$ and $bi$ in the graph.

Return $true$ if the edges of the given graph make up a valid tree, and $false$ otherwise

## Example
### Example 1:
**Input**: $n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]$  
**Output**: $true$

### Example 2:
**Input**: $n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]$  
**Output**: $false$
 
## Constraints:
- $1 \leq n \leq 2000$
- $0 \leq edges.length \leq 5000$
- $edges[i].length == 2$
- $0 \leq a_i, b_i < n$
- $a_i != b_i$
- There are no self-loops or repeated edges