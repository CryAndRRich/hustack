from typing import List

class Solution:
    def permute(self, n: int) -> List[List[int]]:
        res = []
        used = [False] * (n + 1)
        def backtrack(path):
            if len(path) == n:
                res.append(path[:])
                return
            for num in range(1, n + 1):
                if not used[num]:
                    if not path or (path[-1] % 2 != num % 2):
                        used[num] = True
                        path.append(num)
                        backtrack(path)
                        path.pop()
                        used[num] = False
        backtrack([])
        return res
