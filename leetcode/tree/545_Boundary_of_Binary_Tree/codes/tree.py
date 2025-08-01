from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        def isLeaf(node):
            return node and not node.left and not node.right

        def addLeftBoundary(node):
            while node:
                if not isLeaf(node):
                    boundary.append(node.val)
                node = node.left if node.left else node.right

        def addRightBoundary(node):
            stack = []
            while node:
                if not isLeaf(node):
                    stack.append(node.val)
                node = node.right if node.right else node.left
            boundary.extend(reversed(stack))

        def addLeaves(node):
            if not node:
                return
            if isLeaf(node):
                boundary.append(node.val)
                return
            addLeaves(node.left)
            addLeaves(node.right)

        boundary = [root.val] if not isLeaf(root) else []
        addLeftBoundary(root.left)
        addLeaves(root)
        addRightBoundary(root.right)
        return boundary
