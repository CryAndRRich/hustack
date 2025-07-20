from typing import List

class Solution:
    def maximizeSweetness(self, sweetness: List[int], k: int) -> int:
        def canDivide(minSweet):
            count, total = 0, 0
            for s in sweetness:
                total += s
                if total >= minSweet:
                    count += 1
                    total = 0
            return count >= k + 1

        left, right = 1, sum(sweetness)
        while left < right:
            mid = (left + right + 1) // 2
            if canDivide(mid):
                left = mid
            else:
                right = mid - 1
        return left
