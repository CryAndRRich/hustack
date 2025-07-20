import heapq
from typing import List

class Solution:
    def findShortestWay(self, maze: List[List[int]], ball: List[int], hole: List[int]) -> str:
        m, n = len(maze), len(maze[0])
        dirs = [(-1, 0, "u"), (1, 0, "d"), (0, -1, "l"), (0, 1, "r")]
        
        dist = [[(float("inf"), "") for _ in range(n)] for _ in range(m)]
        dist[ball[0]][ball[1]] = (0, "")
        
        pq = [(0, "", ball[0], ball[1])]
        
        while pq:
            d, path, x, y = heapq.heappop(pq)
            if [x, y] == hole:
                return path
            for dx, dy, dirc in dirs:
                nx, ny, steps = x, y, 0
                while 0 <= nx + dx < m and 0 <= ny + dy < n and maze[nx + dx][ny + dy] == 0:
                    nx += dx
                    ny += dy
                    steps += 1
                    if [nx, ny] == hole:
                        break
                if dist[nx][ny][0] > d + steps or (dist[nx][ny][0] == d + steps and dist[nx][ny][1] > path + dirc):
                    dist[nx][ny] = (d + steps, path + dirc)
                    heapq.heappush(pq, (d + steps, path + dirc, nx, ny))
                    
        return "impossible"
