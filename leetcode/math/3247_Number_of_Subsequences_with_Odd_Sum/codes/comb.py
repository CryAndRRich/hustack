from typing import List
import math

MOD = 10**9 + 7

class Solution:
    def subsequenceCount(self, nums: List[int]) -> int:
        even = sum(1 for x in nums if x % 2 == 0)
        odd = len(nums) - even

        res = 0
        for k in range(1, odd + 1, 2):  
            res += math.comb(odd, k) * pow(2, even, MOD)
            res %= MOD
        return res
