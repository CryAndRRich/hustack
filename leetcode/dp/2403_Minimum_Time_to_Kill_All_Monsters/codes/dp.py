from typing import List
from functools import lru_cache
import math

class Solution:
    def minimumTime(self, power: List[int]) -> int:
        n = len(power)

        @lru_cache(None)
        def dp(mask: int, gain: int) -> int:
            if mask == (1 << n) - 1:
                return 0
            ans = float('inf')
            for i in range(n):
                if not (mask & (1 << i)):
                    days = math.ceil(power[i] / gain)
                    ans = min(ans, days + dp(mask | (1 << i), gain + 1))
            return ans

        return dp(0, 1)
