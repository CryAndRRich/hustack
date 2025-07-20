## Description
Given a bipartite graph $G = (U, V, E)$ where $U$ and $V$ are two disjoint sets of vertices and $E$ is the set of edges connecting vertices from $U$ to $V$, find the maximum matching — i.e., the largest set of edges such that no two edges share an endpoint
- The graph is unweighted
- The size of $U$ is $n$, the size of $V$ is $m$, and the number of edges is $E$
- $1 \leq n, m \leq 10^4$, $E \leq 10^5$

**Input:**
- **Line 1:** Two integers $n$ and $m$, the sizes of the two vertex sets
- **Next lines:** Edges represented as pairs of vertices $ (u, v) $ with $ u \in U $, $ v \in V $

**Output:** A single integer - the size of the maximum matching

## Approach 1: Hungarian Algorithm (Kuhn-Munkres)
- Used primarily for **weighted bipartite graphs**, but also applicable for maximum matching on unweighted graphs
- Maintains dual variables (labels) for each vertex in $U$ and $V$
- Repeatedly searches for augmenting paths with minimum slack using DFS or BFS
- Augments matching along the found augmenting path and updates labels

**Time Complexity:** $O(n^3)$  
**Space Complexity:** $O(n^2)$

## Approach 2: Hopcroft-Karp Algorithm
- Designed specifically for **maximum cardinality matching** in bipartite graphs (unweighted)
- Works by finding a maximal set of shortest augmenting paths simultaneously in each phase
- Uses BFS to build layers of the graph and DFS to find vertex-disjoint augmenting paths
- Repeats phases until no more augmenting paths are found

**Time Complexity:** $O(\sqrt{n} \cdot E)$  
**Space Complexity:** $O(n + E)$

## Approach 3: Maximum Flow Reduction (Failed)
- Model the bipartite graph as a flow network:
    - Add a source $S$ connected to all vertices in $U$
    - Add a sink $T$ connected from all vertices in $V$
    - Each edge capacity = 1
- Compute the maximum flow from $S$ to $T$
- The size of the maximum flow equals the maximum bipartite matching
- Can use algorithms like **Edmond-Karp**, **Dinic**, or **Push-Relabel** for max flow

**Time Complexity:** Depends on flow algorithm chosen
- Dinic: $O(\sqrt{n + m} \cdot E)$  
- Edmond-Karp: $O((n + m) \cdot E^2)$  
- Push-Relabel: $O((n + m)^2 \cdot \sqrt{E})$  

**Space Complexity:** $O(n + E)$