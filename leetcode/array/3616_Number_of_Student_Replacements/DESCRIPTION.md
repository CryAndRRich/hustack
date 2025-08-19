## Description
You are given an integer array $ranks$ where $ranks[i]$ represents the rank of the $i^{th}$ student arriving in order. A lower number indicates a better rank.

Initially, the first student is selected by default.

A replacement occurs when a student with a strictly better rank arrives and replaces the current selection.

Return the total number of replacements made.

## Example
### Example 1:
**Input**: $ranks = [4,1,2]$  
**Output**: $1$  
**Explanation**:
- The first student with $ranks[0] = 4$ is initially selected.
- The second student with $ranks[1] = 1$ is better than the current selection, so a replacement occurs.
- The third student has a worse rank, so no replacement occurs.

Thus, the number of replacements is $1$.

### Example 2:
**Input**: $ranks = [2,2,3]$  
**Output**: $0$  
**Explanation**:
- The first student with $ranks[0] = 2$ is initially selected.
- Neither of $ranks[1] = 2$ or $ranks[2] = 3$ is better than the current selection.

Thus, the number of replacements is $0$.
 
## Constraints:
- $1 \leq ranks.length \leq 10^5​​​​​​​$
- $1 \leq ranks[i] \leq 10^5$