from collections import defaultdict
import heapq

def prim(graph, start):
    visited = set()
    heap = [(0, start)] 
    total = 0

    while heap:
        w, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        total += w
        for v, w in graph[u]:
            if v not in visited:
                heapq.heappush(heap, (w, v))

    return total

if __name__ == "__main__":
    n, m = map(int, input().split())
    graph = defaultdict(list)
    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))
    
    res = prim(graph, 1)
    print(res)

