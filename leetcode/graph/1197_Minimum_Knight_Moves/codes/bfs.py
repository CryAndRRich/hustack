from collections import deque

class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        x, y = abs(x), abs(y)
        directions = [(2, 1), (1, 2), (-1, 2), (-2, 1),
                      (-2, -1), (-1, -2), (1, -2), (2, -1)]
        visited = set()
        queue = deque([(0, 0, 0)]) 

        while queue:
            i, j, steps = queue.popleft()
            if (i, j) == (x, y):
                return steps
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if (ni, nj) not in visited and -1 <= ni <= x + 2 and -1 <= nj <= y + 2:
                    visited.add((ni, nj))
                    queue.append((ni, nj, steps + 1))
        return -1
