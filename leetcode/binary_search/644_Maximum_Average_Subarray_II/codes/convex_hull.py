from collections import deque
from itertools import accumulate
from math import inf
from typing import List

class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        n = len(nums)
        prefix = list(accumulate(nums, initial=0))  
        def slope(i: int, j: int) -> float:
            return (prefix[j] - prefix[i]) / (j - i)

        def is_concave(x1: int, x2: int, x3: int) -> bool:
            dx1, dy1 = x2 - x1, prefix[x2] - prefix[x1]
            dx2, dy2 = x3 - x2, prefix[x3] - prefix[x2]
            return dx1 * dy2 < dy1 * dx2

        res = -inf
        dq = deque()

        for j in range(k, n + 1):  
            i = j - k
            while len(dq) >= 2 and is_concave(dq[-2], dq[-1], i):
                dq.pop()
            dq.append(i)

            while len(dq) >= 2 and slope(dq[0], j) <= slope(dq[1], j):
                dq.popleft()

            res = max(res, slope(dq[0], j))

        return res
