import heapq

class Solution:
    def minimumDistance(self, n, edges, s, marked):
        markSet = set(marked)
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
        
        dist = [float('inf')] * n
        dist[s] = 0
        minHeap = [(0, s)]
        
        while minHeap:
            distance, node = heapq.heappop(minHeap)
            if node in markSet:
                return distance
            for nxt, w in adj[node]:
                nd = distance + w
                if nd < dist[nxt]:
                    dist[nxt] = nd
                    heapq.heappush(minHeap, (nd, nxt))
        return -1
