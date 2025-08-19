from typing import List

class Solution:
    def minRunesToAdd(self, n: int, crystals: List[int], flowFrom: List[int], flowTo: List[int]) -> int:
        graph = [[] for _ in range(n)]
        for u, v in zip(flowFrom, flowTo):
            graph[u].append(v)
        
        index = [0 for _ in range(n)]
        lowlink = [0 for _ in range(n)]
        on_stack = [False for _ in range(n)]
        stack = []
        scc_id = [-1 for _ in range(n)]
        idx = 1
        scc_count = 0

        def tarjan(u):
            nonlocal idx, scc_count
            index[u] = idx
            lowlink[u] = idx
            idx += 1
            stack.append(u)
            on_stack[u] = True

            for v in graph[u]:
                if index[v] == 0:
                    tarjan(v)
                    lowlink[u] = min(lowlink[u], lowlink[v])
                elif on_stack[v]:
                    lowlink[u] = min(lowlink[u], index[v])
            
            if lowlink[u] == index[u]:
                while True:
                    node = stack.pop()
                    on_stack[node] = False
                    scc_id[node] = scc_count
                    if node == u:
                        break
                scc_count += 1
        
        for i in range(n):
            if index[i] == 0:
                tarjan(i)

        indegree = [0]*scc_count
        for u in range(n):
            for v in graph[u]:
                if scc_id[u] != scc_id[v]:
                    indegree[scc_id[v]] += 1
        
        crystal_scc = set(scc_id[c] for c in crystals)
        result = 0
        for i in range(scc_count):
            if indegree[i] == 0 and i not in crystal_scc:
                result += 1
        
        return result
