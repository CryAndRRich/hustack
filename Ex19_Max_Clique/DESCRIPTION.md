## Description
Given an undirected graph $G=(V,E)$, find the largest complete subgraph (clique) of $G$

**Input:**
- **Line 1**: Two integers $n$ and $m$ ($1 \le n \le 100$, $1 \le m \le 1000$)
- **Next $m$ lines**: Two integers $u$ and $v$, endpoints of the $i$‑th edge

**Output:**
- **Line 1**: The size of the maximum clique found
- **Line 2**: A list of its vertices, separated by spaces

## Approach 1: Backtrack
### Ver1: Backtrack + Branch and Bound
Perform a depth‑first search, building a clique incrementally and pruning via branch‐and‐bound:
- Maintain a current clique $C$ and a candidate set $R$ of vertices adjacent to all in $C$.
- At each recursive call:
    - If $|C| + |R| \le \text{bestSize}$, prune this branch
    - Otherwise, for each $v \in R$:
        - Add $v$ to $C$
        - Let $R' = R \cap N(v)$ (neighbors of $v$)
        - Recurse on $(C, R')$
        - Remove $v$ from $C$.
    - Track the largest clique found

**Time Complexity**: $O(n.2^n)$  
**Space Complexity**: $O(n+m)$

### Ver2: Backtrack + Branch and Bound + Bitsets
Store neighbors of each vertex in a bitset. Compute $R' = R \& N(v)$ via a single bitwise AND. Pruning as in **Ver1**

**Time Complexity**: $O(2^n \cdot \tfrac{n}{w})$ where $w$ is word size  
**Space Complexity**: $O(\tfrac{n^2}{w})$

## Approach 2: Bron–Kerbosch Algorithm
### Ver1: Basic Bron–Kerbosch
Enumerate maximal cliques recursively using three sets $R,P,X$:
- If $P=\varnothing$ and $X=\varnothing$, report $R$
- Choose a pivot $u\in P\cup X$ to minimize $P\setminus N(u)$
- For each $v\in P\setminus N(u)$:
    - Recurse on $(R\cup{v},P\cap N(v),X\cap N(v))$
    - Remove $v$ from $P$, add to $X$

**Time Complexity**: $O(3^{\tfrac{n}{3}})$  
**Space Complexity**: $O(n^2)$

### Ver2: Bron–Kerbosch + Bitsets
Encode $P,X$ (and neighbor sets) as bitsets. Pivot selection and intersections become bitwise operations

**Time Complexity**: $O(3^{\tfrac{n}{3}} / w)$  
**Space Complexity**: $O(\tfrac{n^2}{w})$