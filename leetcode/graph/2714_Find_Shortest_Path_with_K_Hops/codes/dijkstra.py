from typing import List
import heapq

class Solution:
    def shortestPathWithHops(self, n: int, edges: List[List[int]], s: int, d: int, k: int) -> int:
        g = [[] for _ in range(n)]
        for u, v, w in edges:
            g[u].append((v, w))
            g[v].append((u, w))
        dist = [[float('inf')] * (k + 1) for _ in range(n)]
        dist[s][0] = 0
        pq = [(0, s, 0)]
        while pq:
            t, u, h = heapq.heappop(pq)
            if u == d:
                return t
            if t > dist[u][h]:
                continue
            for v, w in g[u]:
                if t + w < dist[v][h]:
                    dist[v][h] = t + w
                    heapq.heappush(pq, (dist[v][h], v, h))
                if h < k and t < dist[v][h + 1]:
                    dist[v][h + 1] = t
                    heapq.heappush(pq, (dist[v][h + 1], v, h + 1))
        return -1
