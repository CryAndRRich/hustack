from typing import List

class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        res = []

        def dfs(start, target, path):
            for i in range(start, int(target**0.5)+1):
                if target % i == 0:
                    res.append(path + [i, target // i])
                    dfs(i, target // i, path + [i])

        dfs(2, n, [])
        return res
