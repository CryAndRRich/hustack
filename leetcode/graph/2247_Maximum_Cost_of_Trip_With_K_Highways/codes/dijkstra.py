from typing import List
from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def maximumCost(self, n: int, highways: List[List[int]], k: int) -> int:
        if k + 1 > n:
            return -1
        
        graph = defaultdict(list)
        for u, v, cost in highways:
            graph[u].append((v, -cost))
            graph[v].append((u, -cost))
        
        ans = float('inf')
        for start in range(n):
            heap = [(0, start, 1 << start)]
            dist = defaultdict(lambda: float('inf'))
            dist[(start, 1 << start)] = 0
            
            while heap:
                cost, node, mask = heappop(heap)
                if dist[(node, mask)] < cost:
                    continue
                if bin(mask).count('1') == k + 1:
                    ans = min(ans, cost)
                    break
                
                for nxt, c in graph[node]:
                    if (mask & (1 << nxt)) == 0:
                        nmask = mask | (1 << nxt)
                        ncost = cost + c
                        if ncost < dist[(nxt, nmask)]:
                            dist[(nxt, nmask)] = ncost
                            heappush(heap, (ncost, nxt, nmask))
        
        return -1 if ans == float('inf') else -ans
