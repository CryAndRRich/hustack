from typing import List

class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        dp0 = 0  
        dp1 = 0  
        max_len = 0

        for num in nums:
            if num == 1:
                dp0 += 1
                dp1 += 1
            else:
                dp1 = dp0 + 1
                dp0 = 0       

            max_len = max(max_len, dp1)

        return max_len
