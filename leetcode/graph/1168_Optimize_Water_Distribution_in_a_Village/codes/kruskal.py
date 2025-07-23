from typing import List

class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        parent = list(range(n + 1))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[px] = py
            return True

        edges = [[0, i + 1, wells[i]] for i in range(n)] + pipes
        edges.sort(key=lambda x: x[2])

        cost = 0
        for u, v, w in edges:
            if union(u, v):
                cost += w
        return cost
