from bisect import bisect_left
from typing import List

class BIT:
    def __init__(self, n):
        self.n = n
        self.bit = [0] * (n + 2)
    
    def update(self, i, delta):
        i += 1
        while i <= self.n + 1:
            self.bit[i] += delta
            i += i & -i
    
    def query(self, i):
        i += 1
        res = 0
        while i:
            res += self.bit[i]
            i -= i & -i
        return res
    
    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)

class Solution:
    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        n = len(nums)
        index_map = {num: i for i, num in enumerate(nums)}
        bit = BIT(n)
        for i in range(n):
            bit.update(i, 1)
        
        sorted_nums = sorted(nums)
        total = 0
        prev_pos = 0
        for num in sorted_nums:
            pos = index_map[num]
            if pos >= prev_pos:
                ops = bit.range_query(prev_pos, pos)
            else:
                ops = bit.range_query(prev_pos, n - 1) + bit.range_query(0, pos)
            total += ops
            bit.update(pos, -1)
            prev_pos = pos
        return total
