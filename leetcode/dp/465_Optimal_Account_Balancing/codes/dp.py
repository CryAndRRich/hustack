from typing import List

class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        from collections import defaultdict
        n_bal = defaultdict(int)
        for f, t, amt in transactions:
            n_bal[f] -= amt
            n_bal[t] += amt
        balances = [v for v in n_bal.values() if v != 0]
        n = len(balances)
        if n == 0:
            return 0

        total = [0] * (1 << n)
        for mask in range(1 << n):
            for i in range(n):
                if (mask >> i) & 1:
                    total[mask] += balances[i]

        dp = [float('inf')] * (1 << n)
        dp[0] = 0
        for mask in range(1 << n):
            if total[mask] != 0:
                continue
            submask = mask
            while submask:
                dp[mask] = min(dp[mask], dp[mask ^ submask] + bin(submask).count("1") - 1)
                submask = (submask - 1) & mask

        return dp[(1 << n) - 1]
