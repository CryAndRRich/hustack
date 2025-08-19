from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, n: int, m: int, k: int, source: List[int], dest: List[int]) -> int:
        same = 1 if source == dest else 0
        row = 1 if source[0] == dest[0] and source[1] != dest[1] else 0
        col = 1 if source[1] == dest[1] and source[0] != dest[0] else 0
        other = 1 if source[0] != dest[0] and source[1] != dest[1] else 0

        for _ in range(k):
            new_same = (row + col) % MOD
            new_row = ((same * (m - 1)) % MOD + (row * (m - 2)) % MOD + (other * 1) % MOD) % MOD
            new_col = ((same * (n - 1)) % MOD + (col * (n - 2)) % MOD + (other * 1) % MOD) % MOD
            new_other = ((row * (n - 1)) % MOD + (col * (m - 1)) % MOD + (other * (n + m - 4)) % MOD) % MOD

            same, row, col, other = new_same, new_row, new_col, new_other

        return same
