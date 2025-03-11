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

## Result:
<table style="text-align: center;">
    <thead>
        <tr>
            <th style="text-align: center;">File</th>
            <th colspan="2" style="text-align: center;">Test 1</th>
            <th colspan="2" style="text-align: center;">Test 2</th>
            <th colspan="2" style="text-align: center;">Test 3</th>
            <th colspan="2" style="text-align: center;">Test 4</th>
            <th colspan="2" style="text-align: center;">Test 5</th>
        </tr>
        <tr>
            <th></th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
            <th style="text-align: center;">Time (ms)</th>
            <th style="text-align: center;">Memory (MB)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>backtrack.c</td>
            <td>922</td>
            <td>0.965</td>
            <td>6</td>
            <td>0.875</td>
            <td>56</td>
            <td><strong>0.809</strong></td>
            <td>37</td>
            <td><strong>0.715</strong></td>
            <td>54</td>
            <td><strong>0.844</strong></td>
        </tr>
        <tr>
            <td>backtrack.cpp</td>
            <td>820</td>
            <td><strong>0.852</strong></td>
            <td>8</td>
            <td>0.906</td>
            <td>67</td>
            <td>0.926</td>
            <td>37</td>
            <td>0.734</td>
            <td>54</td>
            <td>0.848</td>
        </tr>
        <tr>
            <td>dp.c</td>
            <td>44</td>
            <td>3.348</td>
            <td>6</td>
            <td><strong>0.738</strong></td>
            <td>25</td>
            <td>2.105</td>
            <td><strong>10</strong></td>
            <td>0.992</td>
            <td><strong>14</strong></td>
            <td>1.492</td>
        </tr>
        <tr>
            <td>dp.cpp</td>
            <td><strong>39</strong></td>
            <td>3.492</td>
            <td><strong>5</strong></td>
            <td>0.984</td>
            <td><strong>19</strong></td>
            <td>2.195</td>
            <td><strong>10</strong></td>
            <td>1.18</td>
            <td><strong>14</strong></td>
            <td>1.523</td>
        </tr>
    </tbody>
</table>
