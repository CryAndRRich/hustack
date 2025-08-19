## Description
There is a line chart consisting of $n$ points connected by line segments. You are given a $1$-indexed integer array $y$. The $k^{th}$ point has coordinates $(k, y[k])$. There are no horizontal lines; that is, no two consecutive points have the same y-coordinate.

We can draw an infinitely long horizontal line. Return the maximum number of points of intersection of the line with the chart.

## Example
### Example 1:
**Input**: $y = [1,2,1,2,1,3,2]$  
**Output**: $5$

Example 2:
**Input**: $y = [2,1,3,4,5]$  
**Output**: $2$
 
## Constraints:
- $2 \leq y.length \leq 10^5$
- $1 \leq y[i] \leq 10^9$
- $y[i] != y[i + 1]$ for $i$ in range $[1, n - 1]$