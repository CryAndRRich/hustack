from typing import List

class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        m, k = len(mat1), len(mat1[0])
        n = len(mat2[0])
        result = [[0] * n for _ in range(m)]

        mat1_map = [[(col, val) for col, val in enumerate(row) if val != 0] for row in mat1]
        mat2_map = [[] for _ in range(k)]
        for i in range(k):
            for j in range(n):
                if mat2[i][j] != 0:
                    mat2_map[i].append((j, mat2[i][j]))

        for i in range(m):
            for t, val1 in mat1_map[i]:
                for j, val2 in mat2_map[t]:
                    result[i][j] += val1 * val2
        return result
