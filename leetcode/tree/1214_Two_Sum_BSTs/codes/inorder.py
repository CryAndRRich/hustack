from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def inorder(self, root, res):
        if not root: return
        self.inorder(root.left, res)
        res.append(root.val)
        self.inorder(root.right, res)

    def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
        arr1, arr2 = [], []
        self.inorder(root1, arr1)
        self.inorder(root2, arr2)
        
        i, j = 0, len(arr2) - 1
        while i < len(arr1) and j >= 0:
            s = arr1[i] + arr2[j]
            if s == target:
                return True
            elif s < target:
                i += 1
            else:
                j -= 1
        return False