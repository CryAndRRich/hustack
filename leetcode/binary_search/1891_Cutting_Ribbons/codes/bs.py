from typing import List

class Solution:
    def maxLength(self, ribbons: List[int], k: int) -> int:
        left, right = 1, max(ribbons)
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            count = sum(r // mid for r in ribbons)
            if count >= k:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans
