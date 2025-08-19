from typing import List

MOD = 10**9 + 7

class Solution:
    def numberOfWays(self, n: int, m: int, k: int, source: List[int], dest: List[int]) -> int:
        def multiply(m, a, b, MOD):
            c = [[0] * m for _ in range(m)]
            for i in range(m):
                for j in range(m):
                    for k in range(m):
                        c[i][j] += a[i][k] * b[k][j]
                        c[i][j] %= MOD
            return c

        def multiply1(m, a, b, MOD):
            c = [0] * m
            for i in range(m):
                for j in range(m):
                    c[i] += a[j] * b[i][j]
                    c[i] %= MOD
            return c

        def m_pow(m, matrix, n, MOD , res):
            while n:
                if n % 2:
                    res = multiply1(m, res, matrix, MOD)
                matrix = multiply(m, matrix, matrix, MOD)
                n //= 2
            return res

        matrix = [[0, 1, 1, 0], [n - 1, n - 2, 0, 1], [m - 1, 0, m - 2, 1], [0, m - 1, n - 1, n + m-4]]
        res = [0] * 4 
        if source == dest:
            res[0] = 1
        elif source[0] == dest[0]:
            res[2] = 1
        elif source[1] == dest[1]:
            res[1] = 1
        else:
            res[3] = 1
        ans = m_pow(4, matrix, k, MOD, res)
        return ans[0]
        