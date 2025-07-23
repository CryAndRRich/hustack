import heapq
from typing import List

class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        graph = [[] for _ in range(n + 1)]
        
        for i, cost in enumerate(wells):
            graph[0].append((cost, i + 1))
            graph[i + 1].append((cost, 0))  

        for u, v, w in pipes:
            graph[u].append((w, v))
            graph[v].append((w, u))

        visited = [False] * (n + 1)
        min_heap = [(0, 0)] 
        total_cost = 0

        while min_heap:
            cost, u = heapq.heappop(min_heap)
            if visited[u]:
                continue
            visited[u] = True
            total_cost += cost
            for w, v in graph[u]:
                if not visited[v]:
                    heapq.heappush(min_heap, (w, v))

        return total_cost
