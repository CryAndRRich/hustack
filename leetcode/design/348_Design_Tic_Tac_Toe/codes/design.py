class TicTacToe:
    def __init__(self, n: int):
        self.n = n
        self.rows = [0] * n
        self.cols = [0] * n
        self.diag = 0
        self.anti_diag = 0

    def move(self, row: int, col: int, player: int) -> int:
        mark = 1 if player == 1 else -1
        self.rows[row] += mark
        self.cols[col] += mark
        if row == col:
            self.diag += mark
        if row + col == self.n - 1:
            self.anti_diag += mark

        if abs(self.rows[row]) == self.n or abs(self.cols[col]) == self.n or \
           abs(self.diag) == self.n or abs(self.anti_diag) == self.n:
            return player
        return 0


# Your TicTacToe object will be instantiated and called as such:
# obj = TicTacToe(n)
# param_1 = obj.move(row,col,player)