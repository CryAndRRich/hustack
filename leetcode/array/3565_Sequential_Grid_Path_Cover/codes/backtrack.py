from typing import List

class Solution:
    def findPath(self, grid: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        total = m * n
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        visited = [[False]*n for _ in range(m)]
        result = []

        def backtrack(x, y, count, next_val):
            if visited[x][y]:
                return False
            val = grid[x][y]
            if val != 0 and val != next_val:
                return False
            visited[x][y] = True
            result.append([x, y])
            new_next = next_val + 1 if val == next_val else next_val
            if count == total:
                if new_next == k + 1:
                    return True
                visited[x][y] = False
                result.pop()
                return False
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n:
                    if backtrack(nx, ny, count + 1, new_next):
                        return True
            visited[x][y] = False
            result.pop()
            return False

        starts = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0 or grid[i][j] == 1:
                    starts.append((i, j))
        for sx, sy in starts:
            for i in range(m):
                for j in range(n):
                    visited[i][j] = False
            result.clear()
            if backtrack(sx, sy, 1, 1):
                return result
        return []
