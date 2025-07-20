## Description
Given a **directed graph** $G = (V, E)$ where $V = {1, \dots, N}$ is the set of vertices, and $E$ is the set of $M$ directed arcs. Compute the **number of strongly connected components** (SCCs) in $G$.

**Input**:
- **Line 1:** Two integers $N$ and $M$, the number of nodes and the number of arcs, respectively ($1 \le N \le 10^5,\ 1 \le M \le 10^6$)
- **Next $M$ lines:** Each line contains two integers $u$ and $v$, indicating a directed arc from $u$ to $v$.

**Output**: A single integer - the total number of strongly connected components in $G$.

## Approach 1: Kosaraju’s Algorithm
- Perform a depth-first search (DFS) on the original graph.
- Record the vertices in a list (or stack) in the order of their **finishing times** (i.e., push a vertex onto the stack after all its outgoing edges have been explored).
- Reverse the direction of every arc in $E$ to obtain the **transpose graph** $G^T$.
- Initialize all vertices as unvisited.
- While the stack from the first pass is not empty:
    - Pop a vertex $v$.
    - If $v$ is unvisited, perform a DFS starting from $v$ in $G^T$.
    - Each DFS in $G^T$ identifies one SCC; increment the SCC count by 1.

**Time Complexity:** $O(N + M)$  
- Building the transpose: $O(N + M)$  
- Two DFS traversals: $O(N + M) + O(N + M) = O(N + M)$

**Space Complexity:** $O(N + M)$  
- Adjacency lists for $G$ and $G^T$: $O(N + M)$
- Stack for finishing times: $O(N)$
- Visited marker array: $O(N)$

## Approach 2: Tarjan’s Algorithm
- Maintain an index counter initialized to 0.
- For each unvisited vertex $v$, start a DFS:
    - Assign $index[v] = lowlink[v] = ++counter$.
    - Push $v$ onto a stack and mark it on-stack.
    - For each outgoing edge $v$ to $w$:
        - If $w$ is unvisited, recursively DFS on $w$, then set $lowlink[v] = min(lowlink[v], lowlink[w])$.
        - Else if $w$ is on-stack, $lowlink[v] = min(lowlink[v], index[w])$.
    - After exploring all neighbors of $v$, if $lowlink[v] == index[v]$, then $v$ is the root of an SCC:
        - Pop vertices from the stack until $v$ is removed.
        - All popped vertices form one SCC; increment the SCC count by $1$.

**Time Complexity:** $O(N + M)$  
**Space Complexity:** $O(N + M)$  
- Adjacency lists: $O(N + M)$
- Arrays $index[\,]$, $lowlink[\,]$, and on-stack flags: $O(N)$
- DFS stack (implicit) and explicit node stack: $O(N)$

## Approach 3: Gabow’s Algorithm
- Maintain an index counter.
- For each unvisited vertex $v$, start DFS:
    - Assign $index[v] = ++counter$.
    - Push $v$ onto **stack S** and **stack P**.
    - For each outgoing edge $v$ to $w$:
        - If $index[w]$ is undefined, recursively DFS on $w$ and continue.
        - Else if $w$ is not yet assigned to an SCC, pop from stack P while $index[top(P)] > index[w]$.
    - If the top of stack P is $v$ (i.e., $top(P) == v$), then:
        - Pop vertices from S until $v$ is popped; these form one SCC.
        - Pop $v$ from P and increment SCC count by $1$.

**Time Complexity:** $O(N + M)$  
**Space Complexity:** $O(N + M)$
- Adjacency lists: $O(N + M)$
- Arrays $index[\,]$ and component-assignment flags: $O(N)$
- Two vertex stacks: $O(N)$
