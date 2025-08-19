from typing import List

class SegmentTree:
    def __init__(self, n: int):
        self.n = n
        self.arr = [-1] * (2 * n)
    def update(self, idx: int, val: int):
        idx += self.n
        self.arr[idx] = val
        while idx > 1:
            idx //= 2
            self.arr[idx] = max(self.arr[2 * idx], self.arr[2 * idx + 1])
    def query(self, l: int, r: int) -> int:
        if l > r:
            return -1
        l += self.n
        r += self.n
        res = -1
        while l <= r:
            if l & 1:
                res = max(res, self.arr[l])
                l += 1
            if not (r & 1):
                res = max(res, self.arr[r])
                r -= 1
            l //= 2
            r //= 2
        return res

class Solution:
    def maxProfit(self, prices: List[int], profits: List[int]) -> int:
        n = len(prices)
        mapping = {}
        sorted_prices = sorted(set(prices))
        for i, p in enumerate(sorted_prices):
            mapping[p] = i
        m = len(mapping)
        maxleft = [-1] * n
        maxright = [-1] * n
        segleft = SegmentTree(m)
        segright = SegmentTree(m)
        valleft = [-1] * m
        valright = [-1] * m
        for i in range(n):
            idx = mapping[prices[i]]
            maxleft[i] = segleft.query(0, idx - 1)
            if profits[i] > valleft[idx]:
                valleft[idx] = profits[i]
                segleft.update(idx, profits[i])
        for i in range(n - 1, -1, -1):
            idx = mapping[prices[i]]
            maxright[i] = segright.query(idx + 1, m - 1)
            if profits[i] > valright[idx]:
                valright[idx] = profits[i]
                segright.update(idx, profits[i])
        ans = -1
        for i in range(n):
            if maxleft[i] >= 0 and maxright[i] >= 0:
                ans = max(ans, profits[i] + maxleft[i] + maxright[i])
        return ans
