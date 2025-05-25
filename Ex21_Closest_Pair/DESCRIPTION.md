## Description
Given $N$ points on a plane, find a pair of points with the smallest Euclidean distance between them. No two points coincide and there is a unique closest pair

**Input:**
- **Line $1$:** An integer $N$ $(2 \leq N \leq 100000)$
- **Next $N$ lines:** Two integers $X$ and $Y$ for each point $(|X|, |Y| \leq 10^6)$

**Output:** A single floating‑point number - the minimum distance between the closest pair, rounded to $6$ decimal places

## Approach 1: Divide and Conquer (DnC)
- **Sort** points by $x$‑coordinate into array $Px$
- Recursively split $Px$ into left/right halves until at most $3$ points: brute‑force compute distance among $\leq 3$ points in O(1) for base case
- Let $d_L$ and $d_R$ be the closest distances in left and right halves and $d = min(d_L, d_R)$
- Build a **strip** of points within $d$ of the division line then sort this strip by $y$
- For each point $p$ in strip, compare distances to at most next $6$ points in $y$‑order to update $d$

**Time Complexity:** $O(NlogN)$  
**Space Complexity:** $O(N)$

## Approach 2: KD-Tree
- **Build** a $2‑d$ tree by inserting points one by one, alternating split dimension $(x/y)$ at each depth
- For each point $p$, query the tree for its nearest neighbor among previously inserted points:
    - Traverse to the leaf node where $p$ would be inserted
    - Backtrack, updating the best distance if a closer point is found
    - Prune subtrees whose splitting plane is further than current best
- Insert $p$ into the tree
- Keep track of the global minimum distance

**Time Complexity:** $O(NlogN)$  
**Space Complexity:** $O(N)$
