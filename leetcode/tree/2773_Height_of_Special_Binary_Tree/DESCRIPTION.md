## Description
You are given a $root$, which is the root of a special binary tree with $n$ nodes. The nodes of the special binary tree are numbered from $1$ to $n$. Suppose the tree has k leaves in the following order: $b_1 < b_2 < ... < b_k$.

The leaves of this tree have a special property! That is, for every leaf $b_i$, the following conditions hold:
- The right child of $b_i$ is $b_i + 1$ if $i < k$, and $b_1$ otherwise.
- The left child of $b_i$ is $b_i - 1$ if $i > 1$, and $b_k$ otherwise.

Return the height of the given tree.

Note: The height of a binary tree is the length of the longest path from the root to any other node.

## Example
### Example 1:
**Input**: $root = [1,2,3,null,null,4,5]$  
**Output**: $2$  
**Explanation**: Each leaf's left child is the leaf to its left. Each leaf's right child is the leaf to its right. We can see that the graph has a height of $2$.

### Example 2:
**Input**: $root = [1,2]$  
**Output**: $1$  
**Explanation**: There is only one leaf, so it doesn't have any left or right child. We can see that the graph has a height of $1$.

### Example 3:
**Input**: $root = [1,2,3,null,null,4,null,5,6]$  
**Output**: $3$  
**Explanation**: Each leaf's left child is the leaf to its left. Each leaf's right child is the leaf to its right. We can see that the graph has a height of $3$.

## Constraints:
- $n$ equals number of nodes in the tree
- $2 <= n <= 10^4$
- $1 <= node.val <= n$
- The input is generated such that each $node.val$ is unique.