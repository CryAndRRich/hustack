from typing import List

class Solution:
    def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
        b = [reversed(row) for row in board]
        b = [list(row) for row in zip(*b)]
        m, n = len(b), len(b[0])

        while True:
            crushed = set()
            for i in range(m):
                for j in range(n):
                    v = b[i][j]
                    if v:
                        if i > 1 and v == b[i - 1][j] == b[i - 2][j]:
                            crushed |= {(i, j), (i - 1, j), (i - 2, j)}
                        if j > 1 and v == b[i][j - 1] == b[i][j - 2]:
                            crushed |= {(i, j), (i, j - 1), (i, j - 2)}

            if not crushed:
                return [list(row) for row in zip(*b[::-1])]

            for i, j in crushed:
                b[i][j] = 0

            for row in b:
                row.sort(key=lambda x: x != 0)
