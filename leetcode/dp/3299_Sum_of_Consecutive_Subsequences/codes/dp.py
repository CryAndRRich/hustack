from typing import List

MOD = 10 ** 9 + 7

class Solution:
    def getSum(self, nums: List[int]) -> int:
        n = len(nums)
        
        mx = max(nums)
        dp_inc_before = [0] * (mx + 5)
        dp_dec_before = [0] * (mx + 5)
        left_inc = [0] * n
        left_dec = [0] * n

        for i, x in enumerate(nums):
            left_inc[i] = dp_inc_before[x - 1] if x - 1 >= 0 else 0
            left_dec[i] = dp_dec_before[x + 1] if x + 1 < (mx + 5) else 0
            inc_end = (1 + left_inc[i]) % MOD
            dec_end = (1 + left_dec[i]) % MOD
            dp_inc_before[x] = (dp_inc_before[x] + inc_end) % MOD
            dp_dec_before[x] = (dp_dec_before[x] + dec_end) % MOD

        dp_inc_after = [0] * (mx + 5)
        dp_dec_after = [0] * (mx + 5)
        right_inc = [0] * n
        right_dec = [0] * n

        for i in range(n - 1,  -1,  -1):
            x = nums[i]
            right_inc[i] = dp_inc_after[x + 1] if x + 1 < (mx + 5) else 0
            right_dec[i] = dp_dec_after[x - 1] if x - 1 >= 0 else 0
            inc_start = (1 + right_inc[i]) % MOD
            dec_start = (1 + right_dec[i]) % MOD
            dp_inc_after[x] = (dp_inc_after[x] + inc_start) % MOD
            dp_dec_after[x] = (dp_dec_after[x] + dec_start) % MOD

        ans = 0
        for i, x in enumerate(nums):
            inc_incl = ((1 + left_inc[i]) % MOD) * ((1 + right_inc[i]) % MOD) % MOD
            dec_incl = ((1 + left_dec[i]) % MOD) * ((1 + right_dec[i]) % MOD) % MOD
            total = (inc_incl + dec_incl - 1) % MOD
            ans = (ans + x * total) % MOD
        return ans
