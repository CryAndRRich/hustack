## Description
Given two undirected graphs, $G1 = (V1, E1)$ and $G2 = (V2, E2)$, where:
- $V1 = {1, …, n_1}$ is the set of vertices of $G1$
- $V2 = {1, 2, …, n_2}$ is the set of vertices of $G2$

The graphs are said to be ***isomorphic*** if there exists a bijection $f : V1 → V2$ such that an edge $(u, v)$ exists in $G1$ if and only if the edge $(f(u), f(v))$ exists in $G2$

**Input:**
- **Line 1:** Two positive integers $n_1$ and $m_1$ $(1 \leq n_1 \leq 20, 1 \leq m_1 \leq 40)$ representing the number of vertices and edges in $G1$
- **Next $m_1$ lines:** Each line contains two positive integers $u$ and $v$, denoting an edge in $G1$
- **Line $m_1 + 2$:** Two positive integers $n_2$ and $m_2$ $(1 \leq n_2 \leq 20, 1 \leq m_2 \leq 40)$ representing the number of vertices and edges in $G2$
- **Next $m_2$ lines:** Each line contains two positive integers $u$ and $v$, denoting an edge in $G2$

**Output:** 
$1$ if $G1$ and $G2$ are isomorphic and $0$ otherwise

## Approach 1: Backtracking
This approach attempts to construct a valid bijection between the vertex sets of $G1$ and $G2$ using backtracking. The algorithm:
- Checks if the two graphs have the same number of vertices and edges
- Uses a backtracking strategy to try all possible mappings (subject to pruning based on degree constraints)
- For each candidate mapping, verifies that the edge relationship is preserved

### Time Complexity
- $O(n! \times n^2)$ due to exploring all vertex permutations and checking edge validity for each mapping
- In practice, pruning (e.g., matching only vertices with the same degree) helps reduce the search space

### Space Complexity
- $O(n)$ for storing the current mapping and recursion stack


## Approach 2: VF2 Algorithm
The VF2 algorithm is a state-of-the-art method for graph isomorphism that:
- Uses a systematic state-space search
- Maintains candidate sets and feasibility checks
- Prunes the search space using constraint propagation

This algorithm is often implemented in graph libraries (for example, in Python’s NetworkX library using **GraphMatcher**)

### Time Complexity
- Exponential, but the algorithm usually performs well on graphs of moderate size

### Space Complexity
- Between $O(n)$ and $O(n^2)$ depending on the internal data structures used

## Approach 3: Canonical Labeling
This approach generates a canonical form (a unique string representation) for each graph:
- Convert the graph into an adjacency matrix
- Generate all possible permutations of vertex labels
- For each permutation, produce a string representation of the permuted adjacency matrix
- The lexicographically smallest string is taken as the canonical label

Two graphs are isomorphic if and only if their canonical labels are identical.

### Time Complexity
- $O(n! \times n^2)$ since it checks all vertex permutations

### Space Complexity
- $O(n^2)$ for storing the adjacency matrix and the generated representation strings
