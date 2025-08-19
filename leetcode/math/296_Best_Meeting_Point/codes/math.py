from typing import List

class Solution:
    def minTotalDistance(self, grid: List[List[int]]) -> int:
        def get_coords():
            rows, cols = [], []
            for i in range(len(grid)):
                for j in range(len(grid[0])):
                    if grid[i][j] == 1:
                        rows.append(i)
            for j in range(len(grid[0])):
                for i in range(len(grid)):
                    if grid[i][j] == 1:
                        cols.append(j)
            return rows, cols

        def min_dist(coords):
            coords.sort()
            median = coords[len(coords)//2]
            return sum(abs(c - median) for c in coords)

        rows, cols = get_coords()
        return min_dist(rows) + min_dist(cols)
