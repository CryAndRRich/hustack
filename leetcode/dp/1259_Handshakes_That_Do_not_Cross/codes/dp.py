class Solution:
    def numberOfWays(self, numPeople: int) -> int:
        MOD = 10 ** 9 + 7
        n = numPeople // 2
        dp = [0 for _ in range(n + 1)]
        dp[0] = 1
        
        for i in range(1, n + 1):
            for j in range(i):
                dp[i] = (dp[i] + dp[j] * dp[i - 1 - j]) % MOD
        
        return dp[n]
