from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
        self.max_avg = 0

        def dfs(node):
            if not node:
                return (0, 0)  
            l_sum, l_cnt = dfs(node.left)
            r_sum, r_cnt = dfs(node.right)

            total_sum = l_sum + r_sum + node.val
            total_count = l_cnt + r_cnt + 1
            avg = total_sum / total_count

            self.max_avg = max(self.max_avg, avg)
            return (total_sum, total_count)

        dfs(root)
        return self.max_avg
