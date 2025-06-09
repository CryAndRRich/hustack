import heapq

if __name__ == "__main__":
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        heapq.heappush(edges, (w, u, v))

    parent = [i for i in range(n + 1)]
    rank = [0 for _ in range(n + 1)]
    total = 0
    def find(u):
        while u != parent[u]:
            u = parent[u]
        return u
    
    def union(u, v):
        ru = find(u)
        rv = find(v)
        if ru == rv:
            return False 

        if rank[ru] < rank[rv]:
            parent[ru] = rv
        else:
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
        return True
    
    while edges:
        w, u, v = heapq.heappop(edges)
        if union(u, v):
            total += w

    print(total)
