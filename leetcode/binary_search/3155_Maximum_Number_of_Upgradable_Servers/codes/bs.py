from typing import List

class Solution:
    def maxUpgrades(self, count: List[int], upgrade: List[int], sell: List[int], money: List[int]) -> List[int]:
        def can(mid, c, u, s, m):
            need_sell = c - mid
            total = m + need_sell * s
            return mid * u <= total

        n = len(count)
        result = []
        for i in range(n):
            low, high = 0, count[i]
            best = 0
            while low <= high:
                mid = (low + high) // 2
                if can(mid, count[i], upgrade[i], sell[i], money[i]):
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            result.append(best)
        return result
