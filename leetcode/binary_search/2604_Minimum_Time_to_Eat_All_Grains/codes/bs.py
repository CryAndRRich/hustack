from typing import List

class Solution:
    def minimumTime(self, hens: List[int], grains: List[int]) -> int:
        def can_finish(h: List[int], g: List[int], t: int) -> bool:
            n = len(g)
            r = 0
            for hen in h:
                if hen > grains[r]:
                    if hen - grains[r] > t:
                        return False 
                    diff = hen - grains[r]
                    budget = max((t - diff) // 2, t - 2 * diff)
                else:
                    budget = t 
                while r < n and grains[r] - hen <= budget:
                    r += 1
                if r == n:
                    return True 
            return False

        hens = sorted(hens)
        grains = sorted(grains)
        
        lo, hi = 0, 2000000001
        while lo < hi:
            mid = (lo + hi) // 2
            if can_finish(hens, grains, mid):
                hi = mid 
            else:
                lo = mid + 1
        return lo