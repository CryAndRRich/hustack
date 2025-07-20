from typing import List

class Solution:
    def minCost(self, nums: List[int], costs: List[int]) -> int:
        n = len(nums)
        nse, nge = [n for _ in range(n)], [n for _ in range(n)]
        max_st, min_st = [], []
        
        for i in range(n):
            while max_st and nums[max_st[-1]] <= nums[i]:
                nge[max_st.pop()] = i
            while min_st and nums[min_st[-1]] > nums[i]:
                nse[min_st.pop()] = i
            max_st.append(i)
            min_st.append(i)

        dp = [float('inf') for _ in range(n)]
        dp[0] = 0
        for i in range(n):
            for j in (nse[i], nge[i]):
                if j < n:
                    dp[j] = min(dp[j], dp[i] + costs[j])

        return dp[-1]
