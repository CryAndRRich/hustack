from typing import List
from collections import defaultdict

class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)

        visited = [0] * n

        def dfs(node):
            if visited[node] == 1: 
                return False
            if visited[node] == 2:
                return True
            if not graph[node]:
                return node == destination
            visited[node] = 1
            for nei in graph[node]:
                if not dfs(nei):
                    return False
            visited[node] = 2
            return True

        return dfs(source)
