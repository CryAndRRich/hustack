from typing import List
from collections import defaultdict
from itertools import pairwise

class Solution:
    def maxArea(self, height: int, positions: List[int], directions: str) -> int:
        n = len(positions)

        def can(target: int) -> bool:
            u = directions.count('U')
            mx = sum(positions)
            vertices = defaultdict(int, {0: 0})

            for dir, pos in zip(directions, positions):
                if dir == 'U':
                    vertices[height - pos] -= 1
                    vertices[2 * height - pos] += 1
                else:  
                    vertices[height + pos] -= 1
                    vertices[pos] += 1

            for left, rght in pairwise(sorted(vertices)):
                if mx >= target:
                    return True
                mx += (rght - left) * (2 * u - n)
                u += vertices[rght]

            return mx >= target

        low = sum(positions)
        high = n * height
        ans = low

        while low <= high:
            mid = (low + high) // 2
            if can(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1

        return ans
