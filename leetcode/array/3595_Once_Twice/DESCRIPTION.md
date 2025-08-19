## Description
You are given an integer array #nums#. In this array:
- Exactly one element appears once.
- Exactly one element appears twice.

All other elements appear exactly three times.

Return an integer array of length $2$, where the first element is the one that appears once, and the second is the one that appears twice. Your solution must run in O(n) time and O(1) space.

## Example
### Example 1:
**Input**: $nums = [2,2,3,2,5,5,5,7,7]$  
**Output**: $[3,7]$  
**Explanation**: The element $3$ appears once, and the element $7$ appears twice. The remaining elements each appear three times.

### Example 2:
**Input**: $nums = [4,4,6,4,9,9,9,6,8]$  
**Output**: $[8,6]$  
**Explanation**: The element $8$ appears once, and the element $6$ appears twice. The remaining elements each appear three times.

## Constraints:
- $3 \leq nums.length \leq 10^5$
- $-2^{31} \leq nums[i] \leq 2^{31} - 1$
- $nums.length$ is a multiple of $3$.
- Exactly one element appears once, one element appears twice, and all other elements appear three times.