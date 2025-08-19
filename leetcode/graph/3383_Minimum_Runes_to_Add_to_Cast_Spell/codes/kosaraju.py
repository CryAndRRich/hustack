from collections import defaultdict
from typing import List

class Solution:
    def minRunesToAdd(self, n: int, crystals: List[int], flowFrom: List[int], flowTo: List[int]) -> int:
        adj = defaultdict(list)
        transpose = defaultdict(list)
        for u, v in zip(flowFrom, flowTo):
            adj[u].append(v)
            transpose[v].append(u)
        
        visited = set()
        stack = []
        
        def dfs(u):
            visited.add(u)
            for v in adj[u]:
                if v not in visited:
                    dfs(v)
            stack.append(u)
        
        def dfs_transpose(u, scc):
            visited.add(u)
            scc.append(u)
            for v in transpose[u]:
                if v not in visited:
                    dfs_transpose(v, scc)
        
        for i in range(n):
            if i not in visited:
                dfs(i)
        
        visited.clear()
        sccs = []
        while stack:
            node = stack.pop()
            if node not in visited:
                scc = []
                dfs_transpose(node, scc)
                sccs.append(scc)
        
        scc_map = {}
        for i, scc in enumerate(sccs):
            for node in scc:
                scc_map[node] = i
        
        dag_indegree = [0]*len(sccs)
        edges_set = set()
        for u in range(n):
            for v in adj[u]:
                if scc_map[u] != scc_map[v]:
                    if (scc_map[u], scc_map[v]) not in edges_set:
                        edges_set.add((scc_map[u], scc_map[v]))
                        dag_indegree[scc_map[v]] += 1
        
        crystal_scc = set(scc_map[c] for c in crystals)
        
        count = 0
        for i in range(len(sccs)):
            if dag_indegree[i] == 0 and i not in crystal_scc:
                count += 1
        
        return count
