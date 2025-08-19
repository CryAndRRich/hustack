from typing import List
from collections import defaultdict

class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        if not edges:
            return 0
        
        n = len(edges) + 1
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def dfs(node, parent, dist):
            farthest_node, max_dist = node, dist
            for nxt in graph[node]:
                if nxt != parent:
                    cand_node, cand_dist = dfs(nxt, node, dist + 1)
                    if cand_dist > max_dist:
                        farthest_node, max_dist = cand_node, cand_dist
            return farthest_node, max_dist
        
        node_b, _ = dfs(0, -1, 0)
        _, diameter = dfs(node_b, -1, 0)
        return diameter
