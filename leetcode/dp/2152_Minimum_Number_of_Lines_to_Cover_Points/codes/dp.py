from typing import List

class Solution:
    def minimumLines(self, points: List[List[int]]) -> int:
        n = len(points)
        maskLine = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                mask = 0
                x1, y1 = points[i]
                x2, y2 = points[j]
                for k in range(n):
                    x3, y3 = points[k]
                    if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                        mask |= (1 << k)
                maskLine[i][j] = mask
        
        fullMask = (1 << n) - 1
        dp = [n] * (1 << n)
        dp[0] = 0
        for mask in range(1 << n):
            if dp[mask] >= n:
                continue
            first = -1
            for i in range(n):
                if not (mask & (1 << i)):
                    first = i
                    break
            if first == -1:
                continue
            dp[mask | (1 << first)] = min(dp[mask | (1 << first)], dp[mask] + 1)
            for j in range(n):
                if j != first and not (mask & (1 << j)):
                    newMask = mask | maskLine[first][j]
                    dp[newMask] = min(dp[newMask], dp[mask] + 1)
        return dp[fullMask]
