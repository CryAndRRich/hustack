from typing import List

class Solution:
    def minDistance(self, height: int, width: int, tree: List[int], squirrel: List[int], nuts: List[List[int]]) -> int:
        total = 0
        max_saving = float("-inf")

        for nut in nuts:
            dist_tree = abs(nut[0] - tree[0]) + abs(nut[1] - tree[1])
            dist_squirrel = abs(nut[0] - squirrel[0]) + abs(nut[1] - squirrel[1])
            total += 2 * dist_tree
            max_saving = max(max_saving, dist_tree - dist_squirrel)

        return total - max_saving
