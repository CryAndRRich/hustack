def are_isomorphic(n1, n2, edges1, edges2):
    if n1 != n2:
        return False

    adj1 = [[False for _ in range(n1)] for _ in range(n1)]
    adj2 = [[False for _ in range(n2)] for _ in range(n2)]
    
    for u, v in edges1:
        adj1[u - 1][v - 1] = True
        adj1[v - 1][u - 1] = True

    for u, v in edges2:
        adj2[u - 1][v - 1] = True
        adj2[v - 1][u - 1] = True
    
    deg1 = [sum(1 for j in range(n1) if adj1[i][j]) for i in range(n1)]
    deg2 = [sum(1 for j in range(n2) if adj2[i][j]) for i in range(n2)]
    
    order = list(range(n1))
    order.sort(key=lambda x: deg1[x], reverse=True)
    
    mp = [-1 for _ in range(n1)]
    visited = [False for _ in range(n2)]

    def backtrack(pos):
        if pos == n1:
            return True  
        u = order[pos]
        for v in range(n2):
            if not visited[v] and deg1[u] == deg2[v]:
                mp[u] = v
                visited[v] = True
                valid = True
                for i in range(n1):
                    if mp[i] != -1:
                        if adj1[u][i] != adj2[v][mp[i]]:
                            valid = False
                            break
                if valid and backtrack(pos + 1):
                    return True
                mp[u] = -1
                visited[v] = False
        return False

    return backtrack(0)

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
