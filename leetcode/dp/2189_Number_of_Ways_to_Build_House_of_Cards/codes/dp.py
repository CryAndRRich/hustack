from functools import lru_cache

class Solution:
    def houseOfCards(self, n: int) -> int:
        maxK = (n + 1) // 3

        @lru_cache(None)
        def dp(rem: int, cap: int) -> int:
            if rem == 0:
                return 1
            if cap <= 0:
                return 0
            limit = min(cap, (rem + 1) // 3)
            if limit == 0:
                return 0
            ans = 0
            for k in range(1, limit + 1):
                ans += dp(rem - (3 * k - 1), k - 1)
            return ans
        
        return dp(n, maxK)
