import heapq

def dijkstra(src, n, adj):
    dist = [float("inf") for _ in range(n + 1)]
    parent = [(-1, -1) for _ in range(n + 1)]
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: 
            continue
        for to, c, idx in adj[u]:
            if c < 0: 
                continue
            nd = d + c
            if nd < dist[to]:
                dist[to] = nd
                parent[to] = (u, idx)
                heapq.heappush(pq, (nd, to))
    return dist, parent

if __name__ == "__main__":
    n, m = map(int, input().split())
    adj = [[] for _ in range(n + 1)]
    edges = []
    for i in range(m):
        u, v, c = map(int, input().split())
        edges.append((u, v, c))
        adj[u].append((v, c, i))

    dist1, par1 = dijkstra(1, n, adj)
    if dist1[n] == float("inf"):
        print("NOT_FEASIBLE")
    else:
        potential = dist1[:]
        used = [False for _ in range(m)]
        v = n
        while v != 1:
            u, ei = par1[v]
            used[ei] = True
            v = u
        
        adj2 = [[] for _ in range(n + 1)]
        for i, (u, v, c) in enumerate(edges):
            rc = c + potential[u] - potential[v]
            if not used[i]: 
                adj2[u].append((v, rc, i))
            else:         
                adj2[v].append((u, -rc, i))

        dist2 = [float("inf") for _ in range(n + 1)]
        dist2[1] = 0
        for _ in range(n - 1):
            for u in range(1, n + 1):
                if dist2[u] == float("inf"): 
                    continue
                for to, c, idx in adj2[u]:
                    if dist2[u] + c < dist2[to]:
                        dist2[to] = dist2[u] + c

        if dist2[n] == float("inf"):
            print("NOT_FEASIBLE")
        else:
            ans = 2 * potential[n] + dist2[n]
            print(ans)
