from typing import List
from collections import defaultdict, deque

class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        if not edges:
            return 0
        
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def bfs(start):
            visited = [False] * n
            q = deque([(start, 0)])
            visited[start] = True
            farthest_node, max_dist = start, 0
            while q:
                node, dist = q.popleft()
                if dist > max_dist:
                    max_dist = dist
                    farthest_node = node
                for nxt in graph[node]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        q.append((nxt, dist + 1))
            return farthest_node, max_dist
        
        node_b, _ = bfs(0)
        _, diameter = bfs(node_b)
        return diameter
