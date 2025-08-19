## Description
You are given a positive integer $n$ which is the number of nodes of a $0$-indexed undirected weighted connected graph and a $0$-indexed $2D$ array $edges$ where $edges[i] = [ui, vi, wi]$ indicates that there is an edge between nodes $ui$ and $vi$ with weight $wi$.

You are also given two nodes $s$ and $d$, and a positive integer $k$, your task is to find the shortest path from $s$ to $d$, but you can hop over at most $k$ edges. In other words, make the weight of at most $k$ edges $0$ and then find the shortest path from $s$ to $d$.

Return the length of the shortest path from $s$ to $d$ with the given condition.

## Example
### Example 1:
**Input**: $n = 4, edges = [[0,1,4],[0,2,2],[2,3,6]], s = 1, d = 3, k = 2$  
**Output**: $2$  
**Explanation**: There is only one path from node $1$ to node 3, which is $(1 \rightarrow 0 \rightarrow 2 \rightarrow 3)$ and the length of it is $4 + 2 + 6 = 12$. Now we can make weight of two edges $0$ ($0-1$ and $2-3$), then we have $0 + 2 + 0 = 2$. It can be shown that $2$ is the minimum length of a path we can achieve with the given condition.

### Example 2:
**Input**: $n = 7, edges = [[3,1,9],[3,2,4],[4,0,9],[0,5,6],[3,6,2],[6,0,4],[1,2,4]], s = 4, d = 1, k = 2$  
**Output**: $6$  
**Explanation**: There are $2$ paths from node $4$ to node $1$, which are $(4 \rightarrow 0 \rightarrow 6 \rightarrow 3 \rightarrow 2 \rightarrow 1)$ and $(4 \rightarrow 0 \rightarrow 6 \rightarrow 3 \rightarrow 1)$. The first one has the length $9 + 4 + 2 + 4 + 4 = 23$, and the second one has the length $9 + 4 + 2 + 9 = 24$. Now if we make weight of two edges ($1-3$ and $0-4$) $0$, we get the shortest path with the length $0 + 4 + 2 + 0 = 6$. It can be shown that $6$ is the minimum length of a path we can achieve with the given condition.

### Example 3:
**Input**: $n = 5, edges = [[0,4,2],[0,1,3],[0,2,1],[2,1,4],[1,3,4],[3,4,7]], s = 2, d = 3, k = 1$  
**Output**: $3$  
**Explanation**: There are $4$ paths from node $2$ to node $3$, which are $(2 \rightarrow 1 \rightarrow 3)$, $(2 \rightarrow 0 \rightarrow 1 \rightarrow 3)$, $(2 \rightarrow 1 \rightarrow 0 \rightarrow 4 \rightarrow 3)$ and $(2 \rightarrow 0 \rightarrow 4 \rightarrow 3)$. The first two have the length $4 + 4 = 1 + 3 + 4 = 8$, the third one has the length $4 + 3 + 2 + 7 = 16$ and the last one has the length $1 + 2 + 7 = 10$. Now if we make weight of one edge $(4-3)$ $0$, we get the shortest path with the length $1 + 2 + 0 = 3$. It can be shown that $3$ is the minimum length of a path we can achieve with the given condition.

## Constraints:
- $2 \leq n \leq 500$
- $n - 1 \leq edges.length \leq min(10^4, n \times (n - 1) / 2)$
- $edges[i].length = 3$
- $0 \leq edges[i][0], edges[i][1] \leq n - 1$
- $1 \leq edges[i][2] \leq 10^6$
- $0 \leq s, d, k \leq n - 1$
- $s \neq d$
- The input is generated such that the graph is connected and has no repeated edges or self-loops