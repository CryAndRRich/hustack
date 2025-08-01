from typing import List

class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        def dfs(r, c, direction):
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 1:
                grid[r][c] = 0
                path.append(direction)
                dfs(r + 1, c, "D")
                dfs(r - 1, c, "U")
                dfs(r, c + 1, "R")  
                dfs(r, c - 1, "L") 
                path.append("B") 
        
        shapes = set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    path = []
                    dfs(i, j, "S") 
                    shapes.add(tuple(path))
        return len(shapes)
