from typing import List

class Solution:
    def subsequenceSumOr(self, nums: List[int]) -> int:
        ans = prefix = 0 
        for x in nums: 
            prefix += x
            ans |= x | prefix 
        return ans 