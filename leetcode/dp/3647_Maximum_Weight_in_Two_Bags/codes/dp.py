from typing import List

class Solution:
    def maxWeight(self, weights: List[int], w1: int, w2: int) -> int:
        n = len(weights)
        dp = [[-1] * (w2 + 1) for _ in range(w1 + 1)]
        dp[0][0] = 0
        for w in weights:
            new_dp = [row[:] for row in dp]
            for i in range(w1 + 1):
                for j in range(w2 + 1):
                    if dp[i][j] == -1:
                        continue
                    if i + w <= w1:
                        new_dp[i + w][j] = max(new_dp[i + w][j], dp[i][j] + w)
                    if j + w <= w2:
                        new_dp[i][j + w] = max(new_dp[i][j + w], dp[i][j] + w)
            dp = new_dp
        ans = 0
        for i in range(w1 + 1):
            for j in range(w2 + 1):
                ans = max(ans, dp[i][j])
        return ans
