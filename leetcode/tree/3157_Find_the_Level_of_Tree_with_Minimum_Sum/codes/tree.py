from collections import deque
from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def minimumLevel(self, root: Optional[TreeNode]) -> int:
        q = deque([root])
        level = 1
        min_sum = float('inf')
        ans = 1
        while q:
            s = 0
            for _ in range(len(q)):
                node = q.popleft()
                s += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            if s < min_sum:
                min_sum = s
                ans = level
            level += 1
        return ans
