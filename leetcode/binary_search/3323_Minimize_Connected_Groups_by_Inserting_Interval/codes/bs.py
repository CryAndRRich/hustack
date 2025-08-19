from typing import List

class Solution:
    def minConnectedGroups(self, intervals: List[List[int]], k: int) -> int:
        intervals.sort()
        merged = []
        for s, e in intervals:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        m = len(merged)
        if m == 0:
            return 0

        res = m
        for i in range(m):
            lo, hi, best = i, m - 1, i
            while lo <= hi:
                mid = (lo + hi) // 2
                if merged[mid][0] - merged[i][1] <= k:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            reduction = best - i
            res = min(res, m - reduction)
        return res
