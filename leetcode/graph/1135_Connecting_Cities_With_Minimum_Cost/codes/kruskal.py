from typing import List

class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        parent = list(range(n + 1))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[py] = px
            return True
        
        connections.sort(key=lambda x: x[2])
        cost = 0
        edges_used = 0

        for u, v, w in connections:
            if union(u, v):
                cost += w
                edges_used += 1
                if edges_used == n - 1:
                    return cost
        return -1
