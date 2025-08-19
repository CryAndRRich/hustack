from typing import List

class Solution:
    def equalizeWater(self, buckets: List[int], loss: int) -> float:
        lo, hi = 0.0, max(buckets)
        eps = 1e-6
        factor = (100 - loss) / 100.0

        def can(mid):
            need = 0.0
            have = 0.0
            for b in buckets:
                if b < mid:
                    need += mid - b
                else:
                    have += b - mid
            return have * factor >= need

        while hi - lo > eps:
            mid = (lo + hi) / 2
            if can(mid):
                lo = mid
            else:
                hi = mid
        return lo
