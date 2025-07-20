import heapq
from typing import List

class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        m, n = len(maze), len(maze[0])
        dist = [[float('inf') for _ in range(n)] for _ in range(m)]
        dist[start[0]][start[1]] = 0
        
        heap = [(0 + self.heuristic(start, destination), 0, start[0], start[1])]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while heap:
            _, d, x, y = heapq.heappop(heap)
            if [x, y] == destination:
                return d
            for dx, dy in dirs:
                nx, ny, steps = x, y, 0
                while 0 <= nx + dx < m and 0 <= ny + dy < n and maze[nx + dx][ny + dy] == 0:
                    nx += dx
                    ny += dy
                    steps += 1
                if d + steps < dist[nx][ny]:
                    dist[nx][ny] = d + steps
                    heapq.heappush(heap, (d + steps + self.heuristic([nx, ny], destination), d + steps, nx, ny))
        return -1

    def heuristic(self, pos, dest):
        return abs(pos[0] - dest[0]) + abs(pos[1] - dest[1])
