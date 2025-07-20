import heapq
from typing import List

class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        m, n = len(maze), len(maze[0])
        dist = [[float("inf") for _ in range(n)] for _ in range(m)]
        dist[start[0]][start[1]] = 0
        heap = [(0, start[0], start[1])]

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while heap:
            d, x, y = heapq.heappop(heap)
            if [x, y] == destination:
                return d
            for dx, dy in directions:
                nx, ny, steps = x, y, 0
                while 0 <= nx+dx < m and 0 <= ny+dy < n and maze[nx+dx][ny+dy] == 0:
                    nx += dx
                    ny += dy
                    steps += 1
                if d + steps < dist[nx][ny]:
                    dist[nx][ny] = d + steps
                    heapq.heappush(heap, (d + steps, nx, ny))
                    
        return -1
