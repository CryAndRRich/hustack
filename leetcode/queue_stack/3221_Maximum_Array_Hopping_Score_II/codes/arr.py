from typing import List

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        max_v = nums[-1]
        max_i = n - 1

        idx = [0] * n
        for i in range(n - 2, -1, -1):
            idx[i] = max_i
            if nums[i] > max_v:
                max_v = nums[i]
                max_i = i

        score, i = 0, 0
        while i != n - 1:
            j = idx[i]
            score += (j - i) * nums[j]
            i = j
        return score
