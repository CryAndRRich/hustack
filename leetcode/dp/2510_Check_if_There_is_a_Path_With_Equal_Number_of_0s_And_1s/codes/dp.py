from typing import List
from functools import lru_cache

class Solution:
    def isThereAPath(self, grid: List[List[int]]) -> bool:
        m, n = len(grid), len(grid[0])
        max_diff = m + n
        offset = max_diff 


        @lru_cache(None)
        def dp(i, j, diff):
            val = 1 if grid[i][j] == 0 else -1
            diff += val

            if diff < -offset or diff > offset:
                return False

            if i == 0 and j == 0:
                return diff == 0

            res = False
            if i > 0:
                res |= dp(i - 1, j, diff)
            if j > 0:
                res |= dp(i, j - 1, diff)
            return res

        return dp(m - 1, n - 1, 0)
