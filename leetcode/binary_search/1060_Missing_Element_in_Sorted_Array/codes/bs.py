from typing import List

class Solution:
    def missingElement(self, nums: List[int], k: int) -> int:
        def missing(i):
            return nums[i] - nums[0] - i

        n = len(nums)
        if missing(n - 1) < k:
            return nums[-1] + k - missing(n - 1)

        left, right = 0, n - 1
        while left < right:
            mid = (left + right) // 2
            if missing(mid) < k:
                left = mid + 1
            else:
                right = mid
        return nums[left - 1] + k - missing(left - 1)
