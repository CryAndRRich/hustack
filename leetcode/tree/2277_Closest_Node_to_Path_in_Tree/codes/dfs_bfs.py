from collections import deque, defaultdict
from typing import List

class Solution:
    def closestNode(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        def bfs_dist(src):
            dist = [-1] * n
            q = deque([src])
            dist[src] = 0
            while q:
                node = q.popleft()
                for nei in graph[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        q.append(nei)
            return dist

        def dfs_path(u, target, parent, path):
            path.append(u)
            if u == target:
                return True
            for v in graph[u]:
                if v != parent:
                    if dfs_path(v, target, u, path):
                        return True
            path.pop()
            return False

        res = []
        for start, end, node in query:
            path = []
            dfs_path(start, end, -1, path)
            dist = bfs_dist(node)
            closest = min(path, key=lambda x: dist[x])
            res.append(closest)
        return res
