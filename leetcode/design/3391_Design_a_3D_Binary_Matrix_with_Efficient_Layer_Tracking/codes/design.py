class Matrix3D:

    def __init__(self, n: int):
        self.n = n
        self.matrix = [[[0] * n for _ in range(n)] for _ in range(n)]
        self.count = [0] * n

    def setCell(self, x: int, y: int, z: int) -> None:
        if self.matrix[x][y][z] == 0:
            self.matrix[x][y][z] = 1
            self.count[x] += 1

    def unsetCell(self, x: int, y: int, z: int) -> None:
        if self.matrix[x][y][z] == 1:
            self.matrix[x][y][z] = 0
            self.count[x] -= 1

    def largestMatrix(self) -> int:
        max_count = max(self.count)
        for i in range(self.n - 1, -1, -1):
            if self.count[i] == max_count:
                return i
        return -1
