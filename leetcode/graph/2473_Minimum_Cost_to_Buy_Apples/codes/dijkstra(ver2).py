import heapq

class Solution:
    def minCost(self, n, roads, appleCost, k):
        graph = [[] for _ in range(n)]
        for a, b, c in roads:
            graph[a - 1].append((b - 1, c))
            graph[b - 1].append((a - 1, c))

        result = list(appleCost)
        heap = [(cost, i) for i, cost in enumerate(result)]
        heapq.heapify(heap)

        while heap:
            cost, city = heapq.heappop(heap)
            if cost > result[city]:
                continue
            for nb, w in graph[city]:
                new_cost = result[city] + (k + 1) * w
                if new_cost < result[nb]:
                    result[nb] = new_cost
                    heapq.heappush(heap, (new_cost, nb))
        return result
