def addEdge(u, v, cap, cost, g):
    g[u].append([v, cap, cost, len(g[v])])
    g[v].append([u, 0, -cost, len(g[u]) - 1])

def minCostMaxFlow(s, t, maxf, n, g):
    potential = [0 for _ in range(n + 1)]
    dist = [0 for _ in range(n + 1)]
    pv_v = [0 for _ in range(n + 1)]
    pv_e = [0 for _ in range(n + 1)]
    flow = 0
    cost = 0
    while flow < maxf:
        dist = [float("inf") for _ in range(n + 1)]
        dist[s] = 0
        used = [False for _ in range(n + 1)]
        while True:
            u = -1
            best = float("inf")
            for i in range(1, n + 1):
                if not used[i] and dist[i] < best:
                    best = dist[i]; u = i
            if u == -1: break
            used[u] = True
            for ei, e in enumerate(g[u]):
                to, cap, cst, _ = e
                if cap <= 0: 
                    continue
                nd = dist[u] + cst + potential[u] - potential[to]
                if nd < dist[to]:
                    dist[to] = nd
                    pv_v[to] = u
                    pv_e[to] = ei
        
        if dist[t] == float("inf"): 
            break
        
        for i in range(1, n + 1):
            if dist[i] < float("inf"): 
                potential[i] += dist[i]
        
        addf = maxf - flow
        v = t
        while v != s:
            u = pv_v[v]; ei = pv_e[v]
            addf = min(addf, g[u][ei][1])
            v = u
        flow += addf
        cost += addf * potential[t]
        v = t
        while v != s:
            u = pv_v[v]; ei = pv_e[v]
            e = g[u][ei]
            e[1] -= addf
            g[v][e[3]][1] += addf
            v = u
    return flow, cost

if __name__ == "__main__":
    n, m = map(int, input().split())
    g = [[] for _ in range(n + 1)]
    for _ in range(m):
        u, v, c = map(int, input().split())
        addEdge(u, v, 1, c, g)
    flow, cost = minCostMaxFlow(1, n, 2, n, g)
    if flow < 2:
        print("NOT_FEASIBLE")
    else:
        print(cost)
