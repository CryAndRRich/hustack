## Description
There are $n$ cities $1, 2, ..., n$. The travel distance from city $i$ to city $j$ is $c(i,j)$, for $i,j = 1, 2, ..., n$. A person wants to find a closed tour that visits each city exactly once (except the starting city). Find the itinerary for that person so that the total travel distance is minimal.

**Input**: 
- Line $1$: A positive integers $n$ $(1 \leq k,n \leq 1000)$
- Line $i+1$ $(i = 1, 2, ..., n)$: contains $i^{th}$ row of the distance matrix

**Output**:
- Line $1$: Write n
- Line $2$: write the sequence of points $x_1, x_2,..., x_n$ (a permutation of $1, 2, ..., n$) of the itinerary found

## Approach 1: Backtrack + Branch and Bound
## Approach 2: DP + Bitmask
## Approach 3: Greedy Randomized Heuristic 
## Approach 4: Simulated Annealing
We iteratively explore the solution space by making small random changes (swaps) to the current tour. If the new tour is shorter, it is accepted; otherwise, it may still be accepted with a probability that decreases over time (controlled by a temperature parameter). This allows the algorithm to escape local minima and explore the global optimum. Moreover, we will add some enhancement such that: Adaptive Reheating and Adaptive Initial Temperature
### Time Complexity:
- In each iteration, a swap is performed and evaluated in $O(n)$ time.
- Typical implementations use $O(n^2)$ time, leading to the total complexity is $O(n^3)$
### Space Complexity:
- The total space complexity is $O(n^2)$
## Approach 5: Hybrid Algorithm
### Ver1: Greedy Randomized Heuristic + Simulated Annealing
First, the Greedy algorithm is applied repeatedly (with a randomized parameter $\alpha$) to generate several candidate tours. The best tour among these is chosen as the initial solution. Then, Simulated Annealing is applied starting from that solution. SA makes random local changes (swaps) and accepts worsening moves with a probability that decreases over time (temperature cooling). This combination often yields a better final solution than either method alone.
- **Time Complexity**: $O(n^2)$
- **Space Complexity**: $O(n^2)$
### Ver2: Greedy Randomized Heuristic + Ant Colony Optimization
In this hybrid method, we first generate a high‐quality solution using a randomized Greedy algorithm. We then “seed” the pheromone matrix by boosting the pheromone values along the edges of the Greedy solution. This helps bias the subsequent ACO search toward promising regions of the search space.
- **Time Complexity**: $O(n^2)$
- **Space Complexity**: $O(n^2)$
### Ver3: Greedy Randomized Heuristic + Ant Colony Optimization + Simulated Annealing
Below is an example of a pipeline multi‐phase implementation that combines three methods:
- Phase 1 (Greedy Phase): A randomized greedy algorithm quickly builds an initial tour.
- Phase 2 (ACO Phase): Using the greedy tour to “seed” the pheromone matrix, an Ant Colony Optimization (ACO) procedure is run for several iterations to search for an improved solution.
- Phase 3 (SA Phase): Finally, Simulated Annealing (SA) is applied starting from the best ACO tour to further refine the solution.

Each phase helps to overcome the weaknesses of the other, and together they can yield a high‐quality tour for the TSP.
- **Time Complexity**: $O(n^2)$
- **Space Complexity**: $O(n^2)$
