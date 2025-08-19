import heapq
class Solution:
    def minCost(self, n: int, roads: list[list[int]], appleCost: list[int], k: int) -> list[int]:
        g = [[] for _ in range(n)]
        for a, b, c in roads:
            g[a - 1].append((b - 1, c))
            g[b - 1].append((a - 1, c))
        ans = [0] * n
        for s in range(n):
            dist = [10**18] * n
            dist[s] = 0
            pq = [(0, s)]
            res = 10**18
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                res = min(res, appleCost[u] + (k + 1) * d)
                for v, w in g[u]:
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
            ans[s] = res
        return ans
