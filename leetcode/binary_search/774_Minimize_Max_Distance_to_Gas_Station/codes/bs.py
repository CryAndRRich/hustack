from typing import List

class Solution:
    def minmaxGasDist(self, stations: List[int], k: int) -> float:
        def is_possible(D):
            needed = 0
            for i in range(len(stations) - 1):
                dist = stations[i+1] - stations[i]
                needed += int(dist / D)
            return needed <= k

        lo, hi = 0, stations[-1] - stations[0]
        eps = 1e-6
        while hi - lo > eps:
            mid = (lo + hi) / 2
            if is_possible(mid):
                hi = mid
            else:
                lo = mid
        return lo
