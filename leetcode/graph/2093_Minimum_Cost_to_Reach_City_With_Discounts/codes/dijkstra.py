from typing import List
from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def minimumCost(self, n: int, highways: List[List[int]], discounts: int) -> int:
        graph = defaultdict(list)
        for u, v, cost in highways:
            graph[u].append((v, cost))
            graph[v].append((u, cost))
        
        dist = [[float('inf')] * (discounts + 1) for _ in range(n)]
        dist[0][discounts] = 0
        heap = [(0, 0, discounts)]  
        
        while heap:
            cost, node, d_left = heappop(heap)
            if node == n - 1:
                return cost
            if dist[node][d_left] < cost:
                continue
            for nxt, c in graph[node]:
                if cost + c < dist[nxt][d_left]:
                    dist[nxt][d_left] = cost + c
                    heappush(heap, (cost + c, nxt, d_left))
                if d_left > 0:
                    c_discount = cost + c // 2
                    if c_discount < dist[nxt][d_left - 1]:
                        dist[nxt][d_left - 1] = c_discount
                        heappush(heap, (c_discount, nxt, d_left - 1))
        
        return -1
