## Description
Given an **undirected connected graph** $G = (V, E)$ where $V = \{1, \dots, N\}$. Each edge $(u, v) \in E$ has weight $w(u, v)$. Compute a **minimum spanning tree** (MST) of $G$, i.e., a subset of edges $T \subseteq E$ that connects all vertices in $V$ with the minimum possible total weight.

- The graph is unweighted in the sense that weights are positive integers, but we only care about ordering by weight, not any additional structure.
- $1 \le N, M \le 10^5$ (up to one hundred thousand vertices and edges).
- Each edge weight $w(u,v)$ is a positive integer.

**Input**:
- **Line 1:** Two integers $N$ and $M$, the number of vertices and the number of edges, respectively ($1 \le N, M \le 10^5$)
- **Next $M$ lines (i = 1 … $M$):** Each line contains three positive integers $u$, $v$, and $w$, indicating an undirected edge $(u, v)$ with weight $w$.

**Output**: A single integer: the total weight of the minimum spanning tree found.

## Approach 1: Kruskal's Algorithm
- Sort all edges in non-decreasing order of weight
- Initialize a Union-Find (Disjoint Set Union) structure over vertices $1 \dots N$
- Iterate through the sorted edges $(u, v, w)$:  
    - If $find(u) != find(v)$, then add $(u, v)$ to the MST and perform $union(u, v)$
    - Otherwise, skip the edge (it would form a cycle)  
- Stop when exactly $N - 1$ edges have been added to the MST

**Time Complexity:** $O(M \log M)$
- Sorting edges: $O(M \log M)$
- Union-Find operations (inverse–Ackermann cost) over $M$ edges: $O(M \cdot \alpha(N))$

**Space Complexity:** $O(N + M)$ to store edges and DSU arrays

## Approach 2: Prim’s Algorithm
- Start from any arbitrary vertex (e.g., vertex 1)
- Maintain a boolean array $in_mst[1..N]$ to mark which vertices are already in the MST
- Use a min-heap (priority queue) keyed by edge weight to store all “crossing” edges $(u, v)$ where $u$ is in the MST and $v$ is not yet in the MST
- Repeatedly extract the minimum-weight edge $(u, v)$
    - If $in_mst[v]$ is false, then add $v$ to the MST, mark $in_mst[v] = true$, and push all edges $(v, x)$ (for $x$ not yet in MST) into the heap
    - Otherwise, discard and continue
- Continue until all $N$ vertices are in the MST or the heap is empty (graph is guaranteed connected, so we end after $N - 1$ additions)

**Time Complexity:** $O(M \log N)$  
- Each vertex enters the MST exactly once
- Each edge is pushed into the heap at most twice (once per endpoint entering MST)
- Heap operations: $O(E \log E)$ in the worst case $=\;O(M \log M)$
- If one assumes $M \approx O(N)$, then $O(M \log N)$

**Space Complexity:** $O(N + M)$ for adjacency lists + heap storage.

## Approach 3: Borůvka’s Algorithm
- Initially, treat each vertex as its own component (so there are $N$ components)
- In each round, for **every component** $C$:  
    - Scan all edges $(u, v)$ where $u \in C$, and pick the minimum-weight edge $(u,v)$ such that $v$ lies in a different component
    - Record this “cheapest outgoing edge” for component $C$
- Add all chosen cheapest edges to the MST simultaneously (union their two endpoint components)
- The number of components decreases by at least a factor of 2 each round, so it takes at most $O(\log N)$ rounds until there is a single component  

**Time Complexity:** $O(M \log N)$ 
  - Each round requires scanning all $M$ edges: $O(M)$
  - Number of rounds is $O(\log N)$ (since each round halves the component count)
  - DSU operations across all rounds cost $O(M \times \alpha(N))$  

**Space Complexity:** $O(N + M)$ for edge lists and DSU arrays

## Approach 4: Reverse-Delete Algorithm (Failed)
- Start with the full set of edges $E$
- Sort all edges in **decreasing** order of weight
- For each edge $(u, v, w)$ in that order:  
    - Temporarily remove $(u,v)$ from the graph
    - Check if the graph remains connected (using a DFS or BFS)
        - If it is still connected, permanently delete $(u, v)$
        - Otherwise, restore $(u, v)$ and add $w$ to the MST weight
- At the end, the edges that remain are exactly those in the MST

**Time Complexity:**  
- Sorting edges: $O(M \log M)$
- For each of $M$ edges, run a DFS/BFS over $N + M$: $O(M \cdot (N + M))$
- **Overall:** $O(M^2 + M N)$, which can be up to $O(M^2)$ in the worst case
- In practice, very slow for large $M$, but conceptually simple  

**Space Complexity:** $O(N + M)$ for the adjacency lists and visited array
