if __name__ == "__main__":
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append((w, u, v))

    parent = [i for i in range(n + 1)]
    rank = [0 for _ in range(n + 1)]

    def find(u):
        while u != parent[u]:
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return False
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        else:
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
        return True

    total = 0
    num_comp = n

    while num_comp > 1:
        cheapest = [None for _ in range(n + 1)]

        for w, u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            if cheapest[ru] is None or cheapest[ru][0] > w:
                cheapest[ru] = (w, u, v)
            if cheapest[rv] is None or cheapest[rv][0] > w:
                cheapest[rv] = (w, u, v)

        for i in range(1, n + 1):
            if cheapest[i] is not None:
                w, u, v = cheapest[i]
                if union(u, v):
                    total += w
                    num_comp -= 1

    print(total)
