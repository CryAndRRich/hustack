from typing import List

class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        m, k = len(mat1), len(mat1[0])
        n = len(mat2[0])
        result = [[0] * n for _ in range(m)]

        for i in range(m):
            for t in range(k):
                if mat1[i][t] != 0:
                    for j in range(n):
                        if mat2[t][j] != 0:
                            result[i][j] += mat1[i][t] * mat2[t][j]
        return result
