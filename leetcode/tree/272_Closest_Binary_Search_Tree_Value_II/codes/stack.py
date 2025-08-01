from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def closestKValues(self, root: Optional[TreeNode], target: float, k: int) -> List[int]:
        def initPredecessorStack(node):
            while node:
                if node.val <= target:
                    pred.append(node)
                    node = node.right
                else:
                    node = node.left

        def initSuccessorStack(node):
            while node:
                if node.val > target:
                    succ.append(node)
                    node = node.left
                else:
                    node = node.right

        def getNextPred():
            node = pred.pop()
            val = node.val
            node = node.left
            while node:
                pred.append(node)
                node = node.right
            return val

        def getNextSucc():
            node = succ.pop()
            val = node.val
            node = node.right
            while node:
                succ.append(node)
                node = node.left
            return val

        pred, succ = [], []
        initPredecessorStack(root)
        initSuccessorStack(root)

        res = []
        for _ in range(k):
            if not pred:
                res.append(getNextSucc())
            elif not succ:
                res.append(getNextPred())
            elif abs(pred[-1].val - target) <= abs(succ[-1].val - target):
                res.append(getNextPred())
            else:
                res.append(getNextSucc())
        return res
