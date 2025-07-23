import heapq
from collections import defaultdict
from typing import List

class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        graph = defaultdict(list)
        for u, v, cost in connections:
            graph[u].append((cost, v))
            graph[v].append((cost, u))
        
        visited = set()
        heap = [(0, 1)]  # (cost, node)
        total = 0
        
        while heap and len(visited) < n:
            cost, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            total += cost
            for nei_cost, nei in graph[node]:
                if nei not in visited:
                    heapq.heappush(heap, (nei_cost, nei))
        
        return total if len(visited) == n else -1
