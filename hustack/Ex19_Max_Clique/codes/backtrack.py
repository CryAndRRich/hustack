ans = -1
best = []

def check(G, v, graph):
    for u in G:
        if v not in graph[u]:
            return False
    return True

def backtrack(G, n, graph):
    global ans, best
    if len(G) > ans:
        ans = len(G)
        best = list(x + 1 for x in G)
    for v in range(n):
        if v not in G and check(G, v, graph):
            G.add(v)
            backtrack(G, n, graph)
            G.remove(v)

if __name__ == "__main__":
    n, m = map(int, input().split())
    graph = [set() for _ in range(n)]
    for _ in range(m):
        u,v = map(int, input().split())
        graph[u - 1].add(v - 1)
        graph[v - 1].add(u - 1)
    for i in range(n):
        backtrack(set([i]), n, graph)
    print(ans)
    print(*best)