## Description
You are given an array $nums$. An array is considered positive if the sum of all numbers in each subarray with more than two elements is positive.

You can perform the following operation any number of times:
- Replace one element in nums with any integer between $-10^{18}$ and $10^{18}$.

Find the minimum number of operations needed to make nums positive.

## Example
### Example 1:$
**Input**: $nums = [-10,15,-12]$  
**Output**: $1$  
**Explanation**: The only subarray with more than $2$ elements is the array itself. The sum of all elements is $(-10) + 15 + (-12) = -7$. By replacing $nums[0]$ with $0$, the new sum becomes $0 + 15 + (-12) = 3$. Thus, the array is now positive.

### Example 2:
**Input**: $nums = [-1,-2,3,-1,2,6]$  
**Output**: $1$

### Example 3:
**Input**: $nums = [1,2,3]$  
**Output**: $0$

## Constraints:
- $3 \leq nums.length \leq 10^5$
- $-10^9 \leq nums[i] \leq 10^9$