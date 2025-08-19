## DEscription
You are given an integer array $weights$ and two integers $w_1$ and $w_2$ representing the maximum capacities of two bags.

Each item may be placed in at most one bag such that:
- Bag $1$ holds at most $w_1$ total weight.
- Bag $2$ holds at most $w_2$ total weight.

Return the maximum total weight that can be packed into the two bags.

## Example
### Example 1:
**Input**: $weights = [1,4,3,2], w_1 = 5, w_2 = 4$  
**Output**: $9$  
**Explanation**:
- Bag $1$: Place $weights[2] = 3$ and $weights[3] = 2$ as $3 + 2 = 5 \leq w_1$
- Bag $2$: Place $weights[1] = 4$ as $4 \leq w_2$

Total weight: $5 + 4 = 9$

### Example 2:
**Input**: $weights = [3,6,4,8], w_1 = 9, w_2 = 7$  
**Output**: $15$  
**Explanation**:
- Bag $1$: Place $weights[3] = 8$ as $8 \leq w_1$
- Bag $2$: Place $weights[0] = 3$ and $weights[2] = 4$ as $3 + 4 = 7 \leq w_2$

Total weight: $8 + 7 = 15$

### Example 3:
**Input**: $weights = [5,7], w_1 = 2, w_2 = 3$  
**Output**: $0$  
**Explanation**: No weight fits in either bag, thus the answer is $0$.

## Constraints:
- $1 \leq weights.length \leq 100$
- $1 \leq weights[i] \leq 100$
- $1 \leq w_1, w_2 \leq 300$