from collections import deque
from typing import List

class Solution:
    def minimumCoins(self, prices: List[int]) -> int:
        n = len(prices)
        dp = [0] * n
        dp[0] = prices[0]
        q = deque([0]) 

        for i in range(1, n):
            dp[i] = dp[q[0]] + prices[i]
            while q and q[0] * 2 + 1 < i:
                q.popleft()
            while q and dp[q[-1]] >= dp[i]:
                q.pop()
            q.append(i)

        return dp[q[0]]
