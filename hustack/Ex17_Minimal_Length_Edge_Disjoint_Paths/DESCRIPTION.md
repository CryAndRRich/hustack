## Description
Given a directed graph $G = (V, E)$ with positive edge‐lengths $c(i,j)$, find **two edge‐disjoint** paths from vertex $1$ to vertex $n$ (i.e. no common edges) so that the **sum of their lengths** is as small as possible.

**Input:**  
- **Line 1**: two integers $n, m$ $(1 \le n, m \le 30)$  
- **Next $m$ lines**: three integers $u, v, c$ meaning a directed edge $(u,v)$ with cost $c$

**Output:** The minimum total cost of two edge-disjoint $1\to n$ paths, or $NOT_FEASIBLE$ if no such pair exists

## Approach 1: Min-Cost Max-Flow
**Model as a flow network**  
- Each original edge $(u,v)$ becomes a directed arc with  
    - **capacity** = 1  
    - **cost** = $c(u,v)$  
- Source $s=1$, sink $t=n$, required flow = 2

**Successive shortest augmenting paths**  
- Maintain a **potential** array to keep reduced costs non-negative.  
- Run Dijkstra (with cost = reduced cost) to find the cheapest $s\to t$ augmenting path.  
- Augment one unit of flow, update residual graph and potentials.  
- Repeat until you’ve sent 2 units or no more paths

**Answer**  
- If total flow < 2, print $NOT\_FEASIBLE$
- Otherwise, print total cost

**Time Complexity:** $O(E\log V)$  
**Space Complexity:** $O(V + E)$.

## Approach 2: Suurballe Algorithm

**First shortest path**  
- Run Dijkstra from $1$ to $n$ → get path $P_1$ and distances $dist[v]$

**Reweight edges**  
   - For each edge $(u,v)$, set $c'(u,v) = c(u,v) + dist[u] - dist[v]$
   - Now all $c'\ge0$

**Reverse edges on $P_1$**: For each edge of $P_1$, reverse its direction and set cost $-c'(u,v)$.

**Second shortest path**: Run Dijkstra again on this modified graph → get path $P_2$.

**Recover two disjoint paths**  
- Combine $P_1$ and $P_2$, cancel any opposite‐directed edge pairs → yields two edge-disjoint paths in the original graph.
- **Total cost** = original length of $P_1$ + original length of $P_2$.

**Time Complexity:** $O(E\log V)$  
**Space Complexity:** $O(V + E)$.

## Approach 3: Bitmask-DP Enumeration
**Enumerate all simple paths**  
- DFS from 1 to $n$, track a **bitmask** of used edges and the current cost  
- Store each $(mask,cost)$

**Pairwise check**  
- For each stored pair $(mask_i,c_i)$, $(mask_j,c_j)$, if $mask_i\&mask_j = 0$, they’re edge-disjoint  
- Keep the minimum $c_i + c_j$

**Time Complexity:** Worst-case exponential in $m$, but feasible for $m\le30$ with pruning
**Space Complexity:** $O(P)$, where $P$ = number of simple $1\to n$ paths