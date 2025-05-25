from collections import defaultdict

ans = 0
best = None

def bron_kerbosch(R, P, X, graph):
    global ans, best
    if not P and not X:
        if len(R) > ans:
            ans = len(R)
            best = R
        return
    for v in list(P):
        bron_kerbosch(R | {v}, P & graph[v], X & graph[v], graph)
        P.remove(v)
        X.add(v)

if __name__ == "__main__":
    n, m = map(int, input().split())
    graph = defaultdict(set)
    for _ in range(m):
        u, v = map(int, input().split())
        graph[u].add(v)
        graph[v].add(u)

    bron_kerbosch(set(), set(graph.keys()), set(), graph)
    print(ans)
    print(*best)