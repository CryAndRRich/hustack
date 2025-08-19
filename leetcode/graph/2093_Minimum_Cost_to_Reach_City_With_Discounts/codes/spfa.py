from collections import deque, defaultdict

class Solution:
    def minimumCost(self, n: int, highways: List[List[int]], discounts: int) -> int:
        graph = defaultdict(list)
        for u,v,c in highways:
            graph[u].append((v,c))
            graph[v].append((u,c))

        dist = [[float('inf')] * (discounts+1) for _ in range(n)]
        dist[0][discounts] = 0

        in_queue = [[False] * (discounts+1) for _ in range(n)]
        q = deque()
        q.append((0, discounts))
        in_queue[0][discounts] = True

        while q:
            u, d_left = q.popleft()
            in_queue[u][d_left] = False
            for v, cost in graph[u]:
                new_cost = dist[u][d_left] + cost
                if new_cost < dist[v][d_left]:
                    dist[v][d_left] = new_cost
                    if not in_queue[v][d_left]:
                        q.append((v,d_left))
                        in_queue[v][d_left] = True

                if d_left > 0:
                    new_cost_disc = dist[u][d_left] + cost // 2
                    if new_cost_disc < dist[v][d_left-1]:
                        dist[v][d_left-1] = new_cost_disc
                        if not in_queue[v][d_left-1]:
                            q.append((v,d_left-1))
                            in_queue[v][d_left-1] = True

        res = min(dist[n-1])
        return res if res != float('inf') else -1
