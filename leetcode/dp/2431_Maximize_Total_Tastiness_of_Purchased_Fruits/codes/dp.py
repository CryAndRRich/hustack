from functools import lru_cache
from typing import List

class Solution:
    def maxTastiness(self, price: List[int], tastiness: List[int], maxAmount: int, maxCoupons: int) -> int:
        n = len(price)

        @lru_cache(None)
        def dp(i, budget, coupons):
            if i == n:
                return 0
            ans = dp(i + 1, budget, coupons)
            if budget - price[i] >= 0:
                ans = max(ans, dp(i + 1, budget - price[i], coupons) + tastiness[i])
            if coupons > 0 and budget - (price[i] // 2) >= 0:
                ans = max(ans, dp(i + 1, budget - (price[i] // 2), coupons - 1) + tastiness[i])
            return ans
        
        return dp(0, maxAmount, maxCoupons)
