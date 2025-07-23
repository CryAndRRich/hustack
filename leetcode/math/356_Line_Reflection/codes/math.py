from typing import List

class Solution:
    def isReflected(self, points: List[List[int]]) -> bool:
        point_set = set(map(tuple, points))
        min_x = min(x for x, _ in points)
        max_x = max(x for x, _ in points)
        reflect_x = min_x + max_x  

        for x, y in points:
            if (reflect_x - x, y) not in point_set:
                return False
        return True
