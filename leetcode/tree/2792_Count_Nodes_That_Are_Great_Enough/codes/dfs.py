import heapq
from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def countGreatEnoughNodes(self, root: Optional['TreeNode'], k: int) -> int:
        def dfs(node):
            if not node:
                return [], 0
            left_heap, left_count = dfs(node.left)
            right_heap, right_count = dfs(node.right)

            heap = left_heap + right_heap
            heapq.heapify(heap)
            smallest_k = heapq.nsmallest(k, heap)

            count = left_count + right_count
            if len(smallest_k) == k and node.val > smallest_k[-1]:
                count += 1

            heapq.heappush(heap, node.val)
            heap = heapq.nsmallest(k, heap)
            return heap, count

        return dfs(root)[1]
