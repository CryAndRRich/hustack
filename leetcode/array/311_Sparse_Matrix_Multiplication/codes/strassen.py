from typing import List

class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        def add(A, B):
            n = len(A)
            return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]

        def sub(A, B):
            n = len(A)
            return [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]

        def strassen(A, B):
            n = len(A)
            if n == 1:
                return [[A[0][0] * B[0][0]]]

            mid = n // 2
            def split(M):
                return (
                    [row[:mid] for row in M[:mid]],
                    [row[mid:] for row in M[:mid]],
                    [row[:mid] for row in M[mid:]],
                    [row[mid:] for row in M[mid:]]
                )

            A11, A12, A21, A22 = split(A)
            B11, B12, B21, B22 = split(B)

            M1 = strassen(add(A11, A22), add(B11, B22))
            M2 = strassen(add(A21, A22), B11)
            M3 = strassen(A11, sub(B12, B22))
            M4 = strassen(A22, sub(B21, B11))
            M5 = strassen(add(A11, A12), B22)
            M6 = strassen(sub(A21, A11), add(B11, B12))
            M7 = strassen(sub(A12, A22), add(B21, B22))

            C11 = add(sub(add(M1, M4), M5), M7)
            C12 = add(M3, M5)
            C21 = add(M2, M4)
            C22 = add(sub(add(M1, M3), M2), M6)

            new_matrix = [[0] * n for _ in range(n)]
            for i in range(mid):
                for j in range(mid):
                    new_matrix[i][j] = C11[i][j]
                    new_matrix[i][j + mid] = C12[i][j]
                    new_matrix[i + mid][j] = C21[i][j]
                    new_matrix[i + mid][j + mid] = C22[i][j]
            return new_matrix

        def pad(M, size):
            padded = [[0] * size for _ in range(size)]
            for i in range(len(M)):
                for j in range(len(M[0])):
                    padded[i][j] = M[i][j]
            return padded

        m, k, n = len(mat1), len(mat1[0]), len(mat2[0])
        size = 1
        while size < max(m, k, n):
            size *= 2

        A = pad(mat1, size)
        B = pad(mat2, size)
        C = strassen(A, B)
        return [row[:n] for row in C[:m]]
