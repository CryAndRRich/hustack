from bisect import bisect_left
from typing import List

class FenwickTree:
    def __init__(self, size):
        self.tree = [0] * (size + 2)

    def update(self, i, delta):
        while i < len(self.tree):
            self.tree[i] += delta
            i += (i & -i)

    def query(self, i):
        result = 0
        while i > 0:
            result += self.tree[i]
            i -= (i & -i)
        return result

class Solution:
    def kBigIndices(self, nums: List[int], k: int) -> int:
        sorted_unique = sorted(set(nums))
        compressed = [bisect_left(sorted_unique, x) + 1 for x in nums] 
        n = len(nums)
        left_smaller = [0 for _ in range(n)]
        right_smaller = [0 for _ in range(n)]

        bit = FenwickTree(len(sorted_unique))
        for i in range(n):
            left_smaller[i] = bit.query(compressed[i] - 1)
            bit.update(compressed[i], 1)

        bit = FenwickTree(len(sorted_unique))
        for i in reversed(range(n)):
            right_smaller[i] = bit.query(compressed[i] - 1)
            bit.update(compressed[i], 1)

        return sum(1 for i in range(n) if left_smaller[i] >= k and right_smaller[i] >= k)
