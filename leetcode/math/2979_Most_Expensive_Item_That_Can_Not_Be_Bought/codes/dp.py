class Solution:
    def mostExpensiveItem(self, primeOne: int, primeTwo: int) -> int:
        max_val = primeOne * primeTwo
        dp = [False] * (max_val + 1)
        dp[0] = True

        for i in range(1, max_val + 1):
            if i >= primeOne and dp[i - primeOne]:
                dp[i] = True
            if i >= primeTwo and dp[i - primeTwo]:
                dp[i] = True

        for i in range(max_val, -1, -1):
            if not dp[i]:
                return i
