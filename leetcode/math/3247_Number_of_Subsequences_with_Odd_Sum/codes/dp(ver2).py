from typing import List

class Solution:
    def subsequenceCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        even, odd = 0, 0

        for num in nums:
            if num % 2 == 0:
                even = (even * 2 + 1) % MOD
                odd = (odd * 2) % MOD
            else:
                new_even = (even + odd) % MOD
                new_odd = (even + odd + 1) % MOD
                even, odd = new_even, new_odd

        return odd
