from typing import List

class Solution:
    def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
        m, n = len(board), len(board[0])
        changed = True

        while changed:
            changed = False
            for i in range(m):
                for j in range(n - 2):
                    val = abs(board[i][j])
                    if val != 0 and abs(board[i][j + 1]) == val and abs(board[i][j + 2]) == val:
                        board[i][j] = board[i][j + 1] = board[i][j + 2] = -val
                        changed = True
            for i in range(m - 2):
                for j in range(n):
                    val = abs(board[i][j])
                    if val != 0 and abs(board[i + 1][j]) == val and abs(board[i + 2][j]) == val:
                        board[i][j] = board[i + 1][j] = board[i + 2][j] = -val
                        changed = True

            for j in range(n):
                write = m - 1
                for i in reversed(range(m)):
                    if board[i][j] > 0:
                        board[write][j] = board[i][j]
                        write -= 1
                for i in range(write + 1):
                    board[i][j] = 0

        return board
