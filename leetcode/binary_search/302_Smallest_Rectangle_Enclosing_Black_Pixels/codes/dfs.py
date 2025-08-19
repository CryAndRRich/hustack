class Solution:
    def minArea(self, image: list[list[str]], x: int, y: int) -> int:
        m, n = len(image), len(image[0])
        visited = [[False] * n for _ in range(m)]
        self.min_x = self.max_x = x
        self.min_y = self.max_y = y

        def dfs(i, j):
            if i < 0 or i >= m or j < 0 or j >= n:
                return
            if image[i][j] != '1' or visited[i][j]:
                return
            visited[i][j] = True
            self.min_x = min(self.min_x, i)
            self.max_x = max(self.max_x, i)
            self.min_y = min(self.min_y, j)
            self.max_y = max(self.max_y, j)
            dfs(i+1, j)
            dfs(i-1, j)
            dfs(i, j+1)
            dfs(i, j-1)

        dfs(x, y)
        return (self.max_x - self.min_x + 1) * (self.max_y - self.min_y + 1)
