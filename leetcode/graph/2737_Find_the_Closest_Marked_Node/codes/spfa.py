from collections import deque

class Solution:
    def minimumDistance(self, n: int, edges: list[list[int]], s: int, marked: list[int]) -> int:
        graph = [[] for _ in range(n)]
        for u, v, w in edges:
            graph[u].append((v, w))
        dist = [float('inf')] * n
        dist[s] = 0
        q = deque([s])
        in_queue = [False] * n
        in_queue[s] = True
        while q:
            u = q.popleft()
            in_queue[u] = False
            for v, w in graph[u]:
                if dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    if not in_queue[v]:
                        q.append(v)
                        in_queue[v] = True
        ans = min(dist[m] for m in marked)
        return -1 if ans == float('inf') else ans
