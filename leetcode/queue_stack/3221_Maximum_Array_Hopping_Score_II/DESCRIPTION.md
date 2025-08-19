## Description
Given an array $nums$, you have to get the maximum score starting from index $0$ and hopping until you reach the last element of the array.

In each hop, you can jump from index $i$ to an index $j > i$, and you get a score of $(j - i) \times nums[j]$.

Return the maximum score you can get.

## Example
### Example 1:
**Input**: $nums = [1,5,8]$  
**Output**: $16$  
**Explanation**: There are two possible ways to reach the last element
- $0 \rightarrow 1 \rightarrow 2$ with a score of $(1 - 0) \times 5 + (2 - 1) \times 8 = 13$.
- $0 \rightarrow 2$ with a score of $(2 - 0) \times 8 = 16$.

### Example 2:
**Input**: $nums = [4,5,2,8,9,1,3]$  
**Output**: $42$  
**Explanation**: We can do the hopping $0 \rightarrow 4 \rightarrow 6$ with a score of $(4 - 0) \times 9 + (6 - 4) \times 3 = 42$.

## Constraints:
- $2 \leq nums.length \leq 10^5$
- $1 \leq nums[i] \leq 10^5$