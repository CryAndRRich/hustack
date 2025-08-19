from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def heightOfTree(self, root: Optional[TreeNode]) -> int:
        count_map = {}

        def dfs(node):
            if not node:
                return
            count_map[node.val] = count_map.get(node.val, 0) + 1
            if count_map[node.val] > 1:
                return
            dfs(node.left)
            dfs(node.right)

        dfs(root)

        def max_depth(node, depth):
            if not node or count_map[node.val] > 1:
                return depth
            left = max_depth(node.left, depth + 1)
            right = max_depth(node.right, depth + 1)
            return max(left, right)

        return max_depth(root, 0)
