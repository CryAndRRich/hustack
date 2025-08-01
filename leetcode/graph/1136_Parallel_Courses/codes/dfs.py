from typing import List
from collections import defaultdict

class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        graph = defaultdict(list)
        for u, v in relations:
            graph[u].append(v)

        visited = [0] * (n + 1)
        dp = [0] * (n + 1)

        def dfs(node):
            if visited[node] == 1:
                return -1 
            if visited[node] == 2:
                return dp[node]

            visited[node] = 1
            max_depth = 1
            for nei in graph[node]:
                res = dfs(nei)
                if res == -1:
                    return -1
                max_depth = max(max_depth, res + 1)
            visited[node] = 2
            dp[node] = max_depth
            return dp[node]

        res = 0
        for i in range(1, n + 1):
            r = dfs(i)
            if r == -1:
                return -1
            res = max(res, r)
        return res
