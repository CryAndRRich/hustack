import sys
from typing import List

class Solution:
    def minimumDistance(self, n: int, edges: List[List[int]], s: int, marked: List[int]) -> int:
        dist = [sys.maxsize] * n
        dist[s] = 0

        for _ in range(n - 1):
            for u, v, w in edges:
                if dist[u] != sys.maxsize and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        min_dist = min((dist[m] for m in marked), default=sys.maxsize)

        return -1 if min_dist == sys.maxsize else min_dist