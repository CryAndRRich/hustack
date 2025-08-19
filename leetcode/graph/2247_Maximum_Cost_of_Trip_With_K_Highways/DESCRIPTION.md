## Description
A series of highways connect $n$ cities numbered from $0$ to $n - 1$. You are given a $2D$ integer array highways where $highways[i] = [city1_i, city2_i, toll_i]$ indicates that there is a highway that connects $city1_i$ and $city2_i$, allowing a car to go from $city1_i$ to $city2_i$ and vice versa for a cost of $toll_i$.

You are also given an integer $k$. You are going on a trip that crosses exactly $k$ highways. You may start at any city, but you may only visit each city at most once during your trip.

Return the maximum cost of your trip. If there is no trip that meets the requirements, return $-1$.

## Example
### Example 1:
**Input**: $n = 5, highways = [[0,1,4],[2,1,3],[1,4,11],[3,2,3],[3,4,2]], k = 3$  
**Output**: $17$  
**Explanation**:
- One possible trip is to go from $0 \rightarrow 1 \rightarrow 4 \rightarrow 3$. The cost of this trip is $4 + 11 + 2 = 17$.
- Another possible trip is to go from $4 \rightarrow 1 \rightarrow 2 \rightarrow 3$. The cost of this trip is $11 + 3 + 3 = 17$.

It can be proven that $17$ is the maximum possible cost of any valid trip.

### Example 2:
**Input**: $n = 4, highways = [[0,1,3],[2,3,2]], k = 2$  
**Output**: $-1$
**Explanation**: There are no valid trips of length $2$, so return $-1$.
 

Constraints:
- $2 \leq n \leq 15$
- $1 \leq highways.length \leq 50$
- $highways[i].length == 3$
- $0 \leq city1_i, city2_i \leq n - 1$
- $city1_i != city2_i$
- $0 \leq toll_i \leq 100$
- $1 \leq k \leq 50$
- There are no duplicate highways.