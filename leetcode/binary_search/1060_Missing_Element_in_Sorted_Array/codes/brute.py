from typing import List

class Solution:
    def missingElement(self, nums: List[int], k: int) -> int:
        for i in range(1, len(nums)):
            gap = nums[i] - nums[i - 1] - 1
            if k <= gap:
                return nums[i - 1] + k
            k -= gap
        return nums[-1] + k
