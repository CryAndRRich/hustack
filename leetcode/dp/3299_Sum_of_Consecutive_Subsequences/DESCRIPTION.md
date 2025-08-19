## Description
We call an array $arr$ of length $n$ consecutive if one of the following holds:
- $arr[i] - arr[i - 1] == 1$ for all $1 \leq i < n$.
- $arr[i] - arr[i - 1] == -1$ for all $1 \leq i < n$.

The value of an array is the sum of its elements.

Given an array of integers nums, return the sum of the values of all consecutive non-empty subsequences. Since the answer may be very large, return it modulo $10^9 + 7$.

Note that an array of length 1 is also considered consecutive.

## Example
### Example 1:
Input: $nums = [1,2]$  
Output: $6$  
Explanation: The consecutive subsequences are: $[1]$, $[2]$, $[1, 2]$.

### Example 2:
Input: $nums = [1,4,2,3]$  
Output: $31$  
Explanation: The consecutive subsequences are: $[1]$, $[4]$, $[2]$, $[3]$, $[1, 2]$, $[2, 3]$, $[4, 3]$, $[1, 2, 3]$.

## Constraints:
- $1 \leq nums.length \leq 10^5$
- $1 \leq nums[i] \leq 10^5$