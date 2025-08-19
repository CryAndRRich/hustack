class Solution:
    def findGameWinner(self, n: int) -> bool:
        if n == 1:
            return False
        elif n == 2:
            return True
        g = [0] * (n + 1)
        g[0] = 1
        g[1] = 0
        g[2] = 1
        for i in range(3, n + 1):
            g[i] = (g[i - 1] + 1) ^ (g[i - 2] + 1)
        return g[n] != 0
