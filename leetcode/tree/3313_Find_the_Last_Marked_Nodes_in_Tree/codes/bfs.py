from collections import deque
from typing import List

class Solution:
    def lastMarkedNodes(self, edges: List[List[int]]) -> List[int]:
        n = len(edges) + 1
        g = [[] for _ in range(n)]
        for u, v in edges:
            g[u].append(v)
            g[v].append(u)

        def bfs(start):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            far = start
            while q:
                u = q.popleft()
                far = u
                for v in g[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            return far, dist

        f1, _ = bfs(0)
        f2, dist1 = bfs(f1)
        _, dist2 = bfs(f2)

        res = []
        for i in range(n):
            res.append(f1 if dist1[i] > dist2[i] else f2)
        return res
