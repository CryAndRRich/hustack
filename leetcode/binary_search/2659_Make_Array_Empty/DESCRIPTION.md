## Description
You are given an integer array $nums$ containing distinct numbers, and you can perform the following operations until the array is empty:
- If the first element has the smallest value, remove it
- Otherwise, put the first element at the end of the array.

Return an integer denoting the number of operations it takes to make nums empty.

## Example
### Example 1:
**Input**: $nums = [3,4,-1]$  
**Output**: $5$  
**Explanation**: $[4, -1, 3] \rightarrow [-1, 3, 4] \rightarrow [3, 4] \rightarrow [4] \rightarrow []$

### Example 2:
**Input**: $nums = [1,2,4,3]$  
**Output**: $5$  
**Explanation**: $[2, 4, 3] \rightarrow [4, 3] \rightarrow [3, 4] \rightarrow [4] \rightarrow []$

### Example 3:
**Input**: $nums = [1,2,3]$  
**Output**: $3$  
**Explanation**: $[2, 3] \rightarrow [3] \rightarrow []$
 
## Constraints:
- $1 <= nums.length <= 10^5$
- $-10^9 <= nums[i] <= 10^9$
- All values in nums are distinct