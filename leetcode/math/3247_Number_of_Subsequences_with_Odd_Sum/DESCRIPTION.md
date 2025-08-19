## Description
Given an array $nums$, return the number of subsequences with an odd sum of elements.

Since the answer may be very large, return it modulo $10^9 + 7$.

## Example
### Example 1:
Input: $nums = [1,1,1]$  
Output: $4$  
Explanation: The odd-sum subsequences are $[1, \_, \_]$, $[\_, 1, \_]$, $[\_, \_, 1]$, $[1, 1, 1]$

### Example 2:
Input: nums = [1,2,2]
Output: 4
Explanation: The odd-sum subsequences are $[1, \_, \_]$, $[1, \_, 2]$, $[1, 2, \_]$, $[1, 2, 2]$

## Constraints:
- $1 \leq nums.length \leq 10^5$
- $1 \leq nums[i] \leq 10^9$