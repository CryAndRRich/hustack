from collections import defaultdict
from typing import List

class Solution:
    def minOperationsQueries(self, n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
        tree = [[] for _ in range(n)]
        for u, v, w in edges:
            tree[u].append((v, w))
            tree[v].append((u, w))

        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            parent[find(x)] = find(y)

        visited = [False] * n
        ancestor = [0] * n

        freq = [[0] * 27 for _ in range(n)]

        query_map = defaultdict(list)
        ans = [0] * len(queries)
        for i, (u, v) in enumerate(queries):
            query_map[u].append((v, i))
            query_map[v].append((u, i))

        def dfs(u, p):
            visited[u] = True
            ancestor[u] = u
            for v, w in tree[u]:
                if v == p:
                    continue
                freq[v][:] = freq[u][:]
                freq[v][w] += 1
                dfs(v, u)
                union(v, u)
                ancestor[find(u)] = u

            for v, idx in query_map[u]:
                if visited[v]:
                    lca = ancestor[find(v)]
                    count = [freq[u][k] + freq[v][k] - 2 * freq[lca][k] for k in range(27)]
                    ans[idx] = sum(count) - max(count)

        dfs(0, -1)
        return ans
