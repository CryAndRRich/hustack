## Description
There are $n$ passengers $1, 2, ..., n$. The passenger $i$ want to travel from point $i$ to point $i + n$ $(i = 1,2,…,n)$. There is a bus located at point $0$ and has $k$ places for transporting the passengers (it means at any time, there are at most $k$ passengers on the bus). You are given the distance matrix in which $c(i,j)$ is the traveling distance from point $i$ to point $j$ $(i, j = 0,1,…, 2n)$. Compute the shortest route for the bus, serving $n$ passengers and coming back to point $0$.

**Input**: 
- Line $1$: Two positive integers $n$ and $k$ $(1 \leq n \leq 1000, 1 \leq k \leq 50)$
- Line $i+1$ $(i = 1, 2, ..., 2n+1)$: contains $(i-1)^{th}$ row of the distance matrix

**Output**: 
- Line $1$: Write n 
- Line $2$: The sequence of points (pickup and drop-off) of passengers
## Approach 1: Backtrack + Branch and Bound (Failed)
We use a backtracking approach enhanced with branch and bound to prune inefficient routes by recursively constructing routes by deciding at each step whether to pick up a new passenger (if the bus isn’t full) or to drop off a passenger already on board.

### Ver1:
Using a prediction function to compute a lower bound on the total cost by adding the current cost to an estimate (using the minimum outgoing edge from each unvisited node).
- **Time Complexity**: $O(k.(2n)!)$ $(k \leq 1)$
- **Space Complexity**: $O(n)$

### Ver2:
A simpler prediction function: just sums the distances from the current node to every unvisited node. This reduces computational complexity in the heuristic calculation, even though it might provide a less tight bound compared to the original method.

## Approach 2: DP
We define a DP state $dp[i][j]$ where:
- $i$ is the number of pickups done so far
- $j$ is the number of drop-offs done so far

From each state, the code transitions to the next state by either picking up a new passenger (if the bus isn’t full, i.e., $i - j < k$) or by dropping off a passenger (if there are any on board, i.e., $i - j > k$).
It records the moves with auxiliary arrays ($pre_i$, $pre_j$, and $move$) to later reconstruct the optimal route.

This DP approach dramatically reduces the search space compared to the backtracking methods, achieving a polynomial time solution.

- **Time Complexity**: $O(n^2)$
- **Space Complexity**: $O(n^2)$

## Approach 3: Greedy Randomized Heuristic 

### Ver1:
We construct candidate solutions by:
- Building a list of feasible moves (pickups and drop-offs) at each step.
- Using a threshold determined by an $\alpha$ parameter (here $\alpha=0.3$) to select a restricted candidate list.
- Randomly choosing a move from this list.
- Repeating the process to generate a full candidate solution.
- Iterating 1000 times and keeping the best solution found.

This version does not guarantee an optimal solution but efficiently finds a good solution by randomly exploring the solution space in a guided (greedy) manner. It trades off optimality for speed, making it suitable for large instances where exact methods may be computationally expensive.

- **Time Complexity**: $O(n^2)$
- **Space Complexity**: $O(n^2)$

### Ver2: 
In each iteration, $\alpha$ is adjusted (from $0.1$ down to $0.01$) to control the size of the restricted candidate list ($RCL$). This helps diversify the search and balance between greediness and randomness, potentially yielding better solutions over multiple iterations.
