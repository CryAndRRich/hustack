from typing import List

class Solution:
    def findMinimumTime(self, strength: List[int]) -> int:
        n = len(strength)
        INF = 10 ** 9
        
        cost = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                cost[i][j] = 1000000 - (strength[i] + j)//(j+1)
        
        lx = [max(row) for row in cost]
        ly = [0 for _ in range(n)]
        matchX = [-1 for _ in range(n)]
        matchY = [-1 for _ in range(n)]
        
        def bfs(i):
            from collections import deque
            slack = [INF for _ in range(n)]
            slackx = [-1 for _ in range(n)]
            prev = [-1 for _ in range(n)]
            visX = [False for _ in range(n)]
            visY = [False for _ in range(n)]
            queue = deque()
            queue.append(i)
            visX[i] = True
            while True:
                while queue:
                    u = queue.popleft()
                    for v in range(n):
                        if not visY[v]:
                            delta = lx[u] + ly[v] - cost[u][v]
                            if delta == 0:
                                visY[v] = True
                                if matchY[v] == -1:
                                    x = u
                                    y = v
                                    while x != -1:
                                        nxt = matchX[x]
                                        matchX[x] = y
                                        matchY[y] = x
                                        x = prev[x]
                                        y = nxt
                                    return
                                else:
                                    queue.append(matchY[v])
                                    visX[matchY[v]] = True
                                    prev[matchY[v]] = u
                            elif slack[v] > delta:
                                slack[v] = delta
                                slackx[v] = u
                d = INF
                for v in range(n):
                    if not visY[v] and slack[v] < d:
                        d = slack[v]
                for j in range(n):
                    if visX[j]:
                        lx[j] -= d
                    if visY[j]:
                        ly[j] += d
                    else:
                        slack[j] -= d
                for v in range(n):
                    if not visY[v] and slack[v] == 0:
                        visY[v] = True
                        if matchY[v] == -1:
                            x = slackx[v]
                            y = v
                            while x != -1:
                                nxt = matchX[x]
                                matchX[x] = y
                                matchY[y] = x
                                x = prev[x]
                                y = nxt
                            return
                        else:
                            queue.append(matchY[v])
                            visX[matchY[v]] = True
                            prev[matchY[v]] = slackx[v]
        
        for i in range(n):
            bfs(i)
        
        total = 0
        for i in range(n):
            total += cost[i][matchX[i]]
        
        return n * 1000000 - total
