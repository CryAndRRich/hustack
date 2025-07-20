from collections import defaultdict
from typing import List

class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:

        balance = defaultdict(int)
        for f, t, amt in transactions:
            balance[f] -= amt
            balance[t] += amt

        debt = [v for v in balance.values() if v != 0]

        def dfs(start):
            while start < len(debt) and debt[start] == 0:
                start += 1
            if start == len(debt):
                return 0

            res = float("inf")
            for i in range(start + 1, len(debt)):
                if debt[start] * debt[i] < 0:
                    debt[i] += debt[start]
                    res = min(res, 1 + dfs(start + 1))
                    debt[i] -= debt[start]
            return res

        return dfs(0)
