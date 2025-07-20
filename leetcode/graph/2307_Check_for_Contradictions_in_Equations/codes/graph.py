from typing import List

class Solution:
    def checkContradictions(self, equations: List[List[str]], values: List[float]) -> bool:
        parent = {}
        weight = {}

        def find(x):
            if x not in parent:
                parent[x] = x
                weight[x] = 1.0
                return x
            if parent[x] != x:
                orig_parent = parent[x]
                root = find(orig_parent)
                weight[x] *= weight[orig_parent]
                parent[x] = root
            return parent[x]

        def union(x, y, value):
            root_x, root_y = find(x), find(y)
            if root_x == root_y:
                expected = weight[x] / weight[y]
                return abs(expected - value) < 1e-5
            parent[root_x] = root_y
            weight[root_x] = value * weight[y] / weight[x]
            return True

        for (a, b), val in zip(equations, values):
            if not union(a, b, val):
                return True
        return False
