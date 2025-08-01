from collections import defaultdict, deque
from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        columnTable = defaultdict(list)
        queue = deque([(root, 0)])
        minCol = maxCol = 0

        while queue:
            node, col = queue.popleft()
            if node:
                columnTable[col].append(node.val)
                minCol = min(minCol, col)
                maxCol = max(maxCol, col)
                queue.append((node.left, col - 1))
                queue.append((node.right, col + 1))

        return [columnTable[x] for x in range(minCol, maxCol + 1)]
