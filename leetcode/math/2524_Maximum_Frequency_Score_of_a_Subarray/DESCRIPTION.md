## Description
You are given an integer array $nums$ and a positive integer $k$.

The frequency score of an array is the sum of the distinct values in the array raised to the power of their frequencies, taking the sum modulo $10^9 + 7$.

For example, the frequency score of the array $[5,4,5,7,4,4]$ is $(4^3 + 5^2 + 7^1)$ modulo $(10^9 + 7)$ $= 96$.
Return the maximum frequency score of a subarray of size $k$ in nums. You should maximize the value under the modulo and not the actual value.

A subarray is a contiguous part of an array.

## Example
### Example 1:
**Input**: $nums = [1,1,1,2,1,2], k = 3$  
**Output**: $5$  
**Explanation**: The subarray $[2,1,2]$ has a frequency score equal to $5$. It can be shown that it is the maximum frequency score we can have.

### Example 2:
**Input**: $nums = [1,1,1,1,1,1], k = 4$  
**Output**: $1$  
**Explanation**: All the sub-arrays of length $4$ have a frequency score equal to $1$.
 
## Constraints:
- $1 \leq k \leq nums.length \leq 10^5$
- $1 \leq nums[i] \leq 10^6$