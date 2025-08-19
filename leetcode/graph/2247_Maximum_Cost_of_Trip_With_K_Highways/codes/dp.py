from typing import List

class Solution:
    def maximumCost(self, n: int, highways: List[List[int]], k: int) -> int:
        graph = [[] for _ in range(n)]
        for u, v, cost in highways:
            graph[u].append((v, cost))
            graph[v].append((u, cost))

        if k + 1 > n:
            return -1
        
        max_mask = 1 << n
        dp = [[-1] * n for _ in range(max_mask)]

        for i in range(n):
            dp[1 << i][i] = 0
        
        ans = -1
        for mask in range(max_mask):
            count = bin(mask).count('1')
            if count > k + 1:
                continue
            for u in range(n):
                if dp[mask][u] == -1:
                    continue
                if count == k + 1:
                    ans = max(ans, dp[mask][u])
                    continue
                for v, cost in graph[u]:
                    if (mask & (1 << v)) == 0:
                        new_mask = mask | (1 << v)
                        dp[new_mask][v] = max(dp[new_mask][v], dp[mask][u] + cost)
        
        return ans
