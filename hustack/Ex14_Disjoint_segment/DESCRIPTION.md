## Description
Given a set of segments $X = ((a_1, b_1), \dots, (a_n, b_n))$ in which $a_i < b_i$ are coordinates of the segment $i$ on a line, $i = 1,\dots, n$.  Find a subset of $X$ having largest cardinality in which no two segments of the subset intersect

**Input:**
- **Line $1$:** contains a positive integer $n$ $(1 \leq n \leq 100000)$
- **Line $i+1$ $(i = 1, \dots , n)$:** contains $a_i$ and $b_i$ $(0 \leq a_i \leq b_i \leq 1000000)$

**Output:** Number of segments in the solution found

## Approach 1: DP + Binary Search
Sorts the segments by their end coordinate. For each segment $i$:
- It finds the last segment (with index $ind$) that ends at or before the start of segment $i$ (using binary search)
- It then sets a value $candidate = 1+dp[ind+1]$ (if such an interval exists) or just $1$ otherwise.
- Finally, it updates $dp[i+1]$ to be the maximum of $dp[i]$ (not including segment $i$) and the candidate (including it)

**Time Complexity:** $O(nlogn)$

**Space Complexity:** $O(n)$

## Approach 2: Greedy Algorithm
Sorts the segments by their end coordinate. Iterate through the segments and select the current segment if it does not overlap with the last selected segment

**Time Complexity:** $O(n)$

**Space Complexity:** $O(n)$
