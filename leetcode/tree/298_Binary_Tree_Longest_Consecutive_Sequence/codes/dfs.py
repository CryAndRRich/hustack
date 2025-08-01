from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        self.maxLen = 0

        def dfs(node, prev_val, curr_len):
            if not node:
                return
            if node.val == prev_val + 1:
                curr_len += 1
            else:
                curr_len = 1
            self.maxLen = max(self.maxLen, curr_len)
            dfs(node.left, node.val, curr_len)
            dfs(node.right, node.val, curr_len)

        dfs(root, root.val - 1, 0)
        return self.maxLen
