from itertools import permutations

def canonical_label(n, edges):
    adj = [[0 for _ in range(n)] for _ in range(n)]
    for u, v in edges:
        adj[u - 1][v - 1] = 1
        adj[v - 1][u - 1] = 1

    best = None
    for perm in permutations(range(n)):
        rep = []
        for i in perm:
            row = []
            for j in perm:
                row.append(str(adj[i][j]))
            rep.append("".join(row))
        represent = "".join(rep)
        if best is None or represent < best:
            best = represent
    return best

def are_isomorphic(n1, n2, edges1, edges2):
    if n1 != n2:
        return False
    label1 = canonical_label(n1, edges1)
    label2 = canonical_label(n2, edges2)
    return label1 == label2

if __name__ == '__main__':
    n1, m1 = list(map(int, input().split()))
    edges1 = []
    for _ in range(m1):
        u, v = list(map(int, input().split()))
        edges1.append((u, v))

    n2, m2 = list(map(int, input().split()))
    edges2 = []
    for _ in range(m2):
        u, v = list(map(int, input().split()))
        edges2.append((u, v))
    
    ans = 1 if are_isomorphic(n1, n2, edges1, edges2) else 0
    print(ans)
