from typing import List

class Solution:
    def maxWeight(self, weights: List[int], w1: int, w2: int) -> int:
        dp = [0] * (w1 + 1)
        dp[0] = 1
        for w in weights:
            if w > w1 and w > w2:
                continue
            new_dp = dp[:]
            if w <= w1:
                for i in range(w1, w - 1, -1):
                    new_dp[i] |= dp[i - w]
            if w <= w2:
                for i in range(w1 + 1):
                    new_dp[i] |= (dp[i] << w)
            dp = new_dp
        ans = -1
        for i in range(w1 + 1):
            for j in range(w2, -1, -1):
                if (dp[i] >> j) & 1:
                    ans = max(ans, i + j)
                    break
        return ans