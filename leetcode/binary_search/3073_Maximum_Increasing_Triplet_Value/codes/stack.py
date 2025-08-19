from typing import List

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_right = [0] * n
        right_max = 0

        for k in range(n - 1, -1, -1):
            if nums[k] >= right_max:
                right_max = nums[k]
            else:
                max_right[k] = right_max

        max_triplet = 0
        stack = []

        for j in sorted(range(n), key=lambda x: (nums[x], -x)):
            while stack and stack[-1] > j:
                stack.pop()
            if stack and max_right[j] >= 0:
                max_triplet = max(max_triplet, nums[stack[-1]] - nums[j] + max_right[j])
            stack.append(j)

        return max_triplet
