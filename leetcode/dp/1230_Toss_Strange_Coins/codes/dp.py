from typing import List

class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        n = len(prob)
        dp = [[0.0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1.0

        for i in range(1, n + 1):
            for k in range(target + 1):
                dp[i][k] = dp[i - 1][k] * (1 - prob[i - 1])
                if k > 0:
                    dp[i][k] += dp[i - 1][k - 1] * prob[i - 1]
        
        return dp[n][target]
