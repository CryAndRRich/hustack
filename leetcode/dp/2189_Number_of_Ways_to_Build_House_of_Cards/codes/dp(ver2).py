dp = [0] * 501
dp[0] = 1
for row_size in range(2, 501, 3):
    for reachable in range(500, row_size - 1, -1):
        dp[reachable] += dp[reachable - row_size]

class Solution:
    def houseOfCards(self, n: int) -> int:
        return dp[n]
