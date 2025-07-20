from typing import List

class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        if not costs: 
            return 0

        n, k = len(costs), len(costs[0])
        prev_dp = [0 for _ in range(k)]

        for i in range(n):
            curr_dp = [0 for _ in range(k)]

            min1 = min2 = float("inf")
            ind1 = -1

            for j in range(k):
                if prev_dp[j] < min1:
                    min2 = min1
                    min1 = prev_dp[j]
                    ind1 = j
                elif prev_dp[j] < min2:
                    min2 = prev_dp[j]

            for j in range(k):
                if j == ind1:
                    curr_dp[j] = costs[i][j] + min2
                else:
                    curr_dp[j] = costs[i][j] + min1

            prev_dp = curr_dp

        return min(prev_dp)
