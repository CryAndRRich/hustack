## Description
You are given an integer array $degrees$, where $degrees[i]$ represents the desired degree of the $i^{th}$ vertex.

Your task is to determine if there exists an undirected simple graph with exactly these vertex degrees.

A simple graph has no self-loops or parallel edges between the same pair of vertices.

Return true if such a graph exists, otherwise return false.

## Example
### Example 1:
**Input**: $degrees = [3,1,2,2]$  
**Output**: $true$  

### Example 2:
**Input**: $degrees = [1,3,3,1]$  
**Output**: $false$  

## Constraints:
- $1 \leq n == degrees.length \leq 10^​​​​​​​5$
- $0 \leq degrees[i] \leq n - 1$