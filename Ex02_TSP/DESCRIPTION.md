## Description
There are $n$ cities $1, 2, ..., n$. The travel distance from city $i$ to city $j$ is $c(i,j)$, for $i,j = 1, 2, ..., n$.  A person departs from city $1$, visits each city $2, 3, ..., n$ exactly once and comes back to city $1$. Find the itinerary for that person so that the total travel distance is minimal.

**Input**: 
- Line $1$: A positive integers $n$ $(1 \leq k,n \leq 20)$
- Line $i+1$ $(i = 1, 2, ..., n)$: contains $i^{th}$ row of the distance matrix

**Output**: The total travel distance of the optimal itinerary found.

## Approach 1: Backtrack + Branch and Bound
We employ a backtracking approach combined with branch-and-bound to solve the Traveling Salesman Problem (TSP). It recursively explores all possible orders of visiting cities and uses a heuristic to compute a lower bound on the total tour cost. If the current cost plus the heuristic estimate exceeds the best solution found so far, the algorithm prunes that branch, thereby avoiding unnecessary computations. This strategy significantly reduces the number of paths explored compared to a complete brute-force search.

### Time Complexity:
- Worst-case time complexity is $O(n!)$ since every permutation might be explored.
- Branch-and-bound significantly reduces the search space in practice to $O(k.n!)$ $(k \leq 1)$
### Space Complexity:
Recursion stack depth is at most $n$, leading to a space complexity of $O(n)$

## Approach 2: DP + Bitmask
We solve the Traveling Salesman Problem (TSP) using a DP with bitmask technique. 

We define a state $dp[mask][i]$ representing the minimum cost to visit the set of cities indicated by $mask$ and end at city $i$. The recurrence relation is:
$$dp[mask | (1 << j)][j] = min(dp[mask | (1 << j)][j], dp[mask][i] + cost[i][j])$$

We initialize with $dp[1 << 0][0] = 0$ (starting at city $0$) and, after processing all subsets of cities, the final answer is obtained by adding the cost to return from the last city back to the starting city.

### Time Complexity:
- There are $2^n$ possible subsets (bitmask states).
- For each subset, we iterate over up to $n$ cities (the potential last visited city).
- For each such state, we try to transition to $n$ possible next cities.

Overall time complexity: $O(n.2^n)$
### Space Complexity:
We maintain a DP table of size $n.2^n$, leading to a space complexity of $O(n.2^n)$
