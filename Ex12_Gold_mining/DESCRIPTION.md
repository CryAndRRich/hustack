## Problem Description
The Kingdom ALPHA has **n** warehouses of gold located on a straight line and numbered $1, 2, \dots, n$. Each warehouse $i$ contains an amount of gold denoted by $a_i$ (where $a_i$ is a non-negative integer) and is located at coordinate $i$. The King of ALPHA holds a competition for hunters to select a subset of these warehouses such that the **total amount of gold** is maximized. However, there is a constraint: the distance between any two selected warehouses must be **greater than or equal to $L_1$** and **less than or equal to $L_2$**

**Input:**
- **Line 1:** Three integers $n$, $L_1$, and $L_2$ ($1 \leq n \leq 1000000$, $1 \leq L_1 \leq L_2 \leq n$)
- **Line 2:** $n$ integers representing the amounts of gold: $a_1, a_2, \dots, a_n$

**Output:**
- A single integer representing the maximum total amount of gold that can be obtained by selecting a subset of warehouses under the given distance condition

## Approach 1: Backtracking + Branch and Bound (Failed)
This approach explores all possible subsets of warehouses that satisfy the distance constraints and computes the sum of gold for each valid subset
- **Recursive Exploration:** The algorithm recursively decides whether to include a warehouse. When including warehouse $i$, the next possible warehouse to be included must have an index $j$ such that $j - i \geq L_1$ and $j - i \leq L_2$
- **Validation:** At each step, the algorithm ensures that the selected warehouses satisfy the distance condition
- **Result:** It keeps track of the maximum total amount of gold obtained from any valid subset

**Time Complexity:**
- Worst-case exponential due to exploring all possible subsets, approximately $O(2^n)$ in the worst case 
- Pruning based on the distance constraint can reduce the practical search space

**Space Complexity:**
- $O(n)$ for the recursion stack and storing the current subset

## Approach 2: Dynamic Programming
Dynamic Programming (DP) can be used to optimize the solution by breaking the problem into subproblems. We explore several DP-based methods to handle the constraint on the distance between selected warehouses

### Ver1: Pure DP
Let $dp[i]$ be the maximum total gold that can be collected ending at warehouse $i$. For each warehouse $i$, consider all possible previous warehouses $j$ such that $i - j \geq L_1$ and $i - j \leq L_2$. Then, $dp[i] = \max(dp[i], dp[j] + a_i)$. The final answer is the maximum value in the $dp$ array

- **Time Complexity:** $O(n \times (L_2 - L_1 + 1))$, which may be inefficient for large $n$
- **Space Complexity:** $O(n)$ for the DP array

### Ver 2: DP + Queue
Use a deque (queue) to maintain the maximum $dp[j]$ for indices $j$ in the sliding window corresponding to the valid range [$i-L_2$, $i-L_1$]. For each warehouse $i$, the deque helps to quickly get the maximum $dp[j]$ in the allowed window and update $dp[i] = a_i + \text{(maximum from the queue)}$. This yields the optimal value in an efficient manner

- **Time Complexity:** $O(n)$, as each element is added and removed from the deque at most once
- **Space Complexity:** $O(n)$ for the DP array and the deque

### Ver3: DP + Segment Tree
Use a segment tree to perform range maximum queries. For each warehouse $i$, query the segment tree for the maximum $dp[j]$ in the range corresponding to indices $[i - L_2, i - L_1]$. Update $dp[i] = a_i + \text{(result from segment tree)}$. After calculating $dp[i]$, update the segment tree at index $i$ with the new value. The maximum total gold can be obtained from the DP array after processing all warehouses

- **Time Complexity:** Each range query and update operation takes $O(\log n)$, so overall time is $O(n \log n)$
- **Space Complexity:** $O(n)$ for the DP array and $O(n)$ for the segment tree

### Ver4: DP + Binary Indexed Tree
Use a Fenwick Tree (Binary Indexed Tree) to perform efficient range maximum queries. For each warehouse $i$, use the Fenwick Tree to query the maximum $dp[j]$ for indices in the range $[i - L_2, i - L_1]$. Then, set $dp[i] = a_i + \text{(queried maximum)}$. Update the Fenwick Tree at position $i$ with $dp[i]$. The final answer is the maximum value in the DP array

- **Time Complexity:** Each query and update is $O(\log n)$, leading to an overall time of $O(n \log n)$
- **Space Complexity:** $O(n)$ for both the DP array and the Fenwick Tree
