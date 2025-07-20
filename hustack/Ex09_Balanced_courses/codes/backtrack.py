m, n = list(map(int, input().split()))

P = {}
for i in range(1, m + 1):
    for j in list(map(int, input().split()))[1:]:
        if j not in P:
            P[j] = []
        P[j].append(i)

k = int(input()); 
B = [[False for _ in range(n + 1)] for _ in range(n + 1)]
for _ in range(k):
    a, b = list(map(int, input().split()))
    B[a][b] = True
    B[b][a] = True

x = [0 for _ in range(n + 1)]
load = [0 for _ in range(m + 1)]
best = 10e9

def check(v, k):
    for j in range(1, k):
        if B[j][k] and v == x[j]:
            return False
    return True

def solve(k):
    global best
    for v in P[k]:
        if check(v, k):
            x[k] = v
            load[v] += 1
            if k == n:
                maxi = max(load)
                best = min(best, maxi)
            else:
                if load[v] < best:
                    solve(k + 1)
            load[v] -= 1

solve(1)
print(best)