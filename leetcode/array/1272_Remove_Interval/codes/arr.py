from typing import List

class Solution:
    def removeInterval(self, intervals: List[List[int]], toBeRemoved: List[int]) -> List[List[int]]:
        r1, r2 = toBeRemoved
        res = []
        for a, b in intervals:
            if b <= r1 or a >= r2:
                res.append([a, b])
            else:
                if a < r1:
                    res.append([a, min(b, r1)])
                if b > r2:
                    res.append([max(a, r2), b])
        return res
