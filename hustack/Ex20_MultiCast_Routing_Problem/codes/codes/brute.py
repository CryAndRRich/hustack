import heapq
from itertools import combinations

class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]
    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        else:
            self.p[rb] = ra
            if self.r[ra] == self.r[rb]:
                self.r[ra] += 1
        return True

def solve(n, m, s, L, edges):
    best_cost = float("inf")
    indices = range(m)

    for combo in combinations(indices, n - 1):
        total_cost = sum(edges[i][3] for i in combo)
        if total_cost >= best_cost:
            continue

        dsu = DSU(n)
        valid = True
        for i in combo:
            u, v, _, _ = edges[i]
            if not dsu.union(u, v):
                valid = False
                break
        if not valid:
            continue
        root0 = dsu.find(0)
        if any(dsu.find(i) != root0 for i in range(n)):
            continue

        adj = [[] for _ in range(n)]
        for i in combo:
            u, v, t, _ = edges[i]
            adj[u].append((v, t))
            adj[v].append((u, t))

        dist = [float("inf")] * n
        dist[s] = 0
        heap = [(0, s)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u] or d > L:
                continue
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v] and nd <= L:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        if all(dist[i] != float("inf") for i in range(n)):
            best_cost = total_cost

    if best_cost == float("inf"):
        print("NO_SOLUTION")
    else:
        print(best_cost)

if __name__ == "__main__":
    n, m, s, L = map(int, input().split())
    s -= 1 
    edges = []  
    for _ in range(m):
        u, v, t, c = map(int, input().split())
        edges.append((u - 1, v - 1, t, c))
    
    solve(n, m, s, L, edges)