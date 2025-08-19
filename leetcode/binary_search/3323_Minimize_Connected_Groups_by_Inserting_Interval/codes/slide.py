from typing import List

class Solution:
    def minConnectedGroups(self, intervals: List[List[int]], k: int) -> int:
        intervals.sort()
        merged = []
        for s, e in intervals:
            if not merged or s > merged[-1][1]:
                merged.append([s, e])
            else:
                if e > merged[-1][1]:
                    merged[-1][1] = e
        m = len(merged)
        if m == 0:
            return 0
        j = 0
        max_len = 1
        for i in range(m):
            if j < i:
                j = i
            while j + 1 < m and merged[j + 1][0] - merged[i][1] <= k:
                j += 1
            cur = j - i + 1
            if cur > max_len:
                max_len = cur
        return m - max_len + 1
