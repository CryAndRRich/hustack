from typing import List
from collections import defaultdict

class Solution:
    def closestNode(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        LOG = 20
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)

        parent = [[-1]*LOG for _ in range(n)]
        depth = [0]*n

        def dfs(u, p):
            parent[u][0] = p
            for i in range(1, LOG):
                if parent[u][i-1] != -1:
                    parent[u][i] = parent[parent[u][i-1]][i-1]
            for v in graph[u]:
                if v != p:
                    depth[v] = depth[u]+1
                    dfs(v, u)
        
        def lca(u, v):
            if depth[u] < depth[v]:
                u, v = v, u
            for i in reversed(range(LOG)):
                if parent[u][i] != -1 and depth[parent[u][i]] >= depth[v]:
                    u = parent[u][i]
            if u == v:
                return u
            for i in reversed(range(LOG)):
                if parent[u][i] != -1 and parent[u][i] != parent[v][i]:
                    u = parent[u][i]
                    v = parent[v][i]
            return parent[u][0]

        dfs(0, -1)

        def distance(u, v):
            ancestor = lca(u, v)
            return depth[u] + depth[v] - 2 * depth[ancestor]

        def path_nodes(u, v):
            l = lca(u, v)
            res = []
            while u != l:
                res.append(u)
                u = parent[u][0]
            path_rev = []
            while v != l:
                path_rev.append(v)
                v = parent[v][0]
            return res + [l] + path_rev[::-1]

        res = []
        for u, v, node in query:
            path = path_nodes(u, v)
            res.append(min(path, key=lambda x: distance(x, node)))
        return res
