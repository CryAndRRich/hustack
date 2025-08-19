from typing import List
import bisect

class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        right = [0] * n
        max_val = 0
        for i in range(n - 1, -1, -1):
            right[i] = max_val
            max_val = max(max_val, nums[i])
        sorted_list = []
        ans = 0
        for i in range(n):
            if sorted_list and right[i] > nums[i]:
                idx = bisect.bisect_left(sorted_list, nums[i])
                if idx > 0:
                    greatest_left = sorted_list[idx - 1]
                    ans = max(ans, greatest_left - nums[i] + right[i])
            bisect.insort(sorted_list, nums[i])
        return ans
