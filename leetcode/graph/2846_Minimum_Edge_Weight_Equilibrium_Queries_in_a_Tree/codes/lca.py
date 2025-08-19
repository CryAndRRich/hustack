from math import log2
from typing import List

class Solution:
    def minOperationsQueries(self, n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
        tree = [[] for _ in range(n)]
        for u, v, w in edges:
            tree[u].append((v, w))
            tree[v].append((u, w))
        m = int(log2(n)) + 2
        lift = [[-1] * m for _ in range(n)]
        freq = [[0] * 27 for _ in range(n)]
        depth = [-1] * n
        stack = [(0, -1, 0)]
        while stack:
            u, p, d = stack.pop()
            depth[u] = d
            for v, w in tree[u]:
                if v != p:
                    lift[v][0] = u
                    freq[v][:] = freq[u][:]
                    freq[v][w] += 1
                    for j in range(1, m):
                        if lift[v][j-1] == -1:
                            break
                        lift[v][j] = lift[lift[v][j-1]][j-1]
                    stack.append((v, u, d+1))
        def lca(u, v):
            if depth[u] > depth[v]:
                u, v = v, u
            for i in range(m):
                if (depth[v] - depth[u]) >> i & 1:
                    v = lift[v][i]
            if u == v:
                return u
            for i in range(m-1, -1, -1):
                if lift[u][i] != lift[v][i]:
                    u, v = lift[u][i], lift[v][i]
            return lift[u][0]
        ans = []
        for u, v in queries:
            k = lca(u, v)
            count = [freq[u][w] + freq[v][w] - 2 * freq[k][w] for w in range(27)]
            ans.append(sum(count) - max(count))
        return ans
