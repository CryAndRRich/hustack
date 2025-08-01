from typing import Optional

#Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        self.count = 0

        def is_unival(node):
            if not node:
                return True
            
            left = is_unival(node.left)
            right = is_unival(node.right)

            if not left or not right:
                return False

            if node.left and node.left.val != node.val:
                return False
            if node.right and node.right.val != node.val:
                return False

            self.count += 1
            return True

        is_unival(root)
        return self.count
