## Description
You are given two integers $height$ and $width$ representing a garden of size height x width. You are also given:
- an array $tree$ where $tree = [tree_r, tree_c]$ is the position of the tree in the garden,
- an array $squirrel$ where $squirrel = [squirrel_r, squirrel_c]$ is the position of the squirrel in the garden,
- and an array $nuts$ where $nuts[i] = [nut_{i_r}, nut_{i_c}]$ is the position of the ith nut in the garden

The squirrel can only take at most one nut at one time and can move in four directions: up, down, left, and right, to the adjacent cell.

Return the minimal distance for the squirrel to collect all the nuts and put them under the tree one by one. The distance is the number of moves.

## Example
### Example 1:
**Input**: $height = 5, width = 7, tree = [2,2], squirrel = [4,4], nuts = [[3,0], [2,5]]$ 
**Output**: $12$  
**Explanation**: The squirrel should go to the nut at [2, 5] first to achieve a minimal distance

### Example 2:
**Input**: $height = 1, width = 3, tree = [0,1], squirrel = [0,0], nuts = [[0,2]]$  
**Output**: $3$
 
## Constraints:
- $1 \leq height, width \leq 100$
- $tree.length == 2$
- $squirrel.length == 2$
- $1 \leq nuts.length \leq 5000$
- $nuts[i].length == 2$
- $0 \leq tree_r, squirrel_r, nut_{i_r} \leq height$
- $0 \leq tree_c, squirrel_c, nut_{i_c} \leq width$