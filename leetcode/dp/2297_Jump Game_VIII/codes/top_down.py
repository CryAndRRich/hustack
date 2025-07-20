from functools import cache
from typing import List

class Solution:
    def minCost(self, nums: List[int], costs: List[int]) -> int:
        n = len(nums)
        nse, nge = [n] * n, [n] * n
        max_st, min_st = [], []

        for i in range(n):
            while max_st and nums[max_st[-1]] <= nums[i]:
                nge[max_st.pop()] = i
            while min_st and nums[min_st[-1]] > nums[i]:
                nse[min_st.pop()] = i
            max_st.append(i)
            min_st.append(i)

        @cache
        def dp(i):
            if i == n:
                return float('inf')
            if i == n - 1:
                return 0 if i == 0 else costs[i]
            cost = 0 if i == 0 else costs[i]
            return cost + min(dp(nse[i]), dp(nge[i]))
        
        return dp(0)
