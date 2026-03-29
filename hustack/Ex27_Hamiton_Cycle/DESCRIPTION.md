## Description
Given an **undirected graph** $G = (V,E)$ where $V = \{1, \dots, n\}$ is the set of vertices, and $E$ is the set of $m$ edges. Write a program to check if $G$ is a **Hamiltonian graph** (i.e., it contains a Hamiltonian cycle, which is a closed loop that visits every vertex exactly once and returns to the starting vertex).

**Input**:
- **Line 1:** A positive integer $T$, the number of test cases (graphs).
- **Subsequent lines:** Information about the $T$ graphs, each formatted as follows:
  - **Line 1:** Two integers $n$ and $m$, representing the number of nodes and the number of edges, respectively (typically $1 \le n \le 20$ due to the NP-Complete nature of the problem).
  - **Next $m$ lines:** Each line contains two integers $u$ and $v$, indicating an undirected edge between $u$ and $v$.

**Output**: 
- For each graph, write $1$ on a new line if the corresponding graph is a Hamiltonian graph, and write $0$ otherwise.

## Approach 1: Backtracking (Depth-First Search)
- Represent the graph using an adjacency matrix or adjacency list.
- Use a path array to store the current sequence of vertices and a visited boolean array to keep track of the vertices already included in the path.
- Start the search from an arbitrary vertex (e.g., vertex $1$). Mark it as visited and add it to the first position of the path.
- For the current vertex at position pos, iterate through all vertices $v$:
  - If $v$ is adjacent to the current vertex and has not been visited, recursively attempt to add $v$ to the next position pos + 1.
  - Mark $v$ as visited.
  - If the recursive call returns true, propagate the success upwards.
  - If the recursive call returns false, **backtrack**: remove $v$ from the path and mark it as unvisited.
- **Base Case:** If pos == n (all vertices are in the path), check if there is an edge connecting the last vertex in the path back to the starting vertex. If there is, a Hamiltonian cycle exists; return true.

**Time Complexity:** $O(n!)$ 
- In the worst-case scenario, the algorithm explores all permutations of the vertices to find a valid cycle.

**Space Complexity:** $O(n^2)$ 
- Adjacency matrix: $O(n^2)$
- Recursion call stack and arrays (path, visited): $O(n)$

## Approach 2: Bitmask Dynamic Programming (Held-Karp Algorithm)
- Represent the set of visited vertices using an integer bitmask. The $i$-th bit of the mask is set to $1$ if vertex $i$ has been visited, and $0$ otherwise.
- Define a boolean DP table $dp[mask][u]$ which is true if there exists a simple path visiting exactly the set of vertices represented by mask, and ending at vertex $u$.
- Convert the 1-based vertex indices to 0-based indexing for easier bit manipulation ($0$ to $n-1$).
- **Initialization:** Fix an arbitrary starting vertex, say vertex $0$. Set $dp[1][0] = true$ (binary 00...001 represents only vertex $0$ visited). All other states are false.
- **Transitions:** Iterate over all possible masks from $1$ to $2^n - 1$:
  - Skip masks where the starting vertex $0$ is not visited (i.e., the $0$-th bit is $0$).
  - For each vertex $u$ in the graph, if $dp[mask][u]$ is true, it means we have a valid path ending at $u$.
  - Iterate through all unvisited neighbors $v$ of $u$ (where the $v$-th bit in mask is $0$ and there is an edge between $u$ and $v$).
  - Turn on the $v$-th bit in the mask to form a new_mask ($mask \mid (1 \ll v)$) and set $dp[new\_mask][v] = true$.
- **Result Extraction:** After filling the DP table, check the state where all vertices are visited (i.e., $full\_mask = 2^n - 1$). Iterate through all vertices $u$. If $dp[full\_mask][u]$ is true and there is an edge back to the starting vertex $0$, then the graph is Hamiltonian.

**Time Complexity:** $O(n^2 \cdot 2^n)$ 
- There are $2^n$ possible subsets (masks) of vertices.
- For each subset, we iterate through $n$ possible ending vertices $u$ and check transitions to $n$ possible next vertices $v$.

**Space Complexity:** $O(n \cdot 2^n)$
- DP table $dp[2^n][n]$: $O(n \cdot 2^n)$
- Adjacency matrix: $O(n^2)$