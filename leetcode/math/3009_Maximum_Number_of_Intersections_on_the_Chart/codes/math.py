from typing import List
from collections import defaultdict

class Solution:
    def maxIntersectionCount(self, y: List[int]) -> int:
        n = len(y)
        events = defaultdict(int)
        for i in range(1, n):
            s = 2 * y[i - 1]
            if i == n - 1:
                e = 2 * y[i]
            else:
                e = 2 * y[i] + (-1 if y[i] > y[i - 1] else 1)
            lo = s if s <= e else e
            hi = e if s <= e else s
            events[lo] += 1
            events[hi + 1] -= 1
        cur = 0
        ans = 0
        for key in sorted(events.keys()):
            cur += events[key]
            if cur > ans:
                ans = cur
        return ans
