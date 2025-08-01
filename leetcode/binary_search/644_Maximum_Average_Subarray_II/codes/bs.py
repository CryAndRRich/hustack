from typing import List

class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        def is_valid(mid):
            prefix = [0]
            for num in nums:
                prefix.append(prefix[-1] + num - mid)
            
            min_prev = 0
            for i in range(k, len(prefix)):
                if prefix[i] - min_prev >= 0:
                    return True
                min_prev = min(min_prev, prefix[i - k + 1])
            return False

        left, right = min(nums), max(nums)
        epsilon = 1e-5

        while right - left > epsilon:
            mid = (left + right) / 2
            if is_valid(mid):
                left = mid
            else:
                right = mid

        return left
