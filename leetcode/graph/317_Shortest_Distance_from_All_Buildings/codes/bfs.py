from collections import deque
from typing import List

class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        totalDist = [[0]*n for _ in range(m)]
        reachCount = [[0]*n for _ in range(m)]
        buildings = [(i, j) for i in range(m) for j in range(n) if grid[i][j] == 1]
        
        for bx, by in buildings:
            visited = [[False]*n for _ in range(m)]
            q = deque([(bx, by, 0)])
            while q:
                x, y, d = q.popleft()
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < m and 0 <= ny < n and not visited[nx][ny] and grid[nx][ny] == 0:
                        visited[nx][ny] = True
                        totalDist[nx][ny] += d + 1
                        reachCount[nx][ny] += 1
                        q.append((nx, ny, d+1))
        
        minDist = float("inf")
        totalBuildings = len(buildings)
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 and reachCount[i][j] == totalBuildings:
                    minDist = min(minDist, totalDist[i][j])
        
        return minDist if minDist != float("inf") else -1
