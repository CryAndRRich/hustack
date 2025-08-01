/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int nextPowerOfTwo(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

int** allocateMatrix(int size) {
    int** mat = (int**)malloc(size * sizeof(int*));
    for (int i = 0; i < size; ++i)
        mat[i] = (int*)calloc(size, sizeof(int));
    return mat;
}

void freeMatrix(int** mat, int size) {
    for (int i = 0; i < size; ++i) free(mat[i]);
    free(mat);
}

void add(int** A, int** B, int** C, int size) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            C[i][j] = A[i][j] + B[i][j];
}

void subtract(int** A, int** B, int** C, int size) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            C[i][j] = A[i][j] - B[i][j];
}

void strassen(int** A, int** B, int** C, int size) {
    if (size == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    int newSize = size / 2;
    int** A11 = allocateMatrix(newSize), **A12 = allocateMatrix(newSize);
    int** A21 = allocateMatrix(newSize), **A22 = allocateMatrix(newSize);
    int** B11 = allocateMatrix(newSize), **B12 = allocateMatrix(newSize);
    int** B21 = allocateMatrix(newSize), **B22 = allocateMatrix(newSize);
    int** M1 = allocateMatrix(newSize), **M2 = allocateMatrix(newSize);
    int** M3 = allocateMatrix(newSize), **M4 = allocateMatrix(newSize);
    int** M5 = allocateMatrix(newSize), **M6 = allocateMatrix(newSize);
    int** M7 = allocateMatrix(newSize);
    int** AResult = allocateMatrix(newSize), **BResult = allocateMatrix(newSize);

    for (int i = 0; i < newSize; ++i)
        for (int j = 0; j < newSize; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }

    add(A11, A22, AResult, newSize);
    add(B11, B22, BResult, newSize);
    strassen(AResult, BResult, M1, newSize);

    add(A21, A22, AResult, newSize);
    strassen(AResult, B11, M2, newSize);

    subtract(B12, B22, BResult, newSize);
    strassen(A11, BResult, M3, newSize);

    subtract(B21, B11, BResult, newSize);
    strassen(A22, BResult, M4, newSize);

    add(A11, A12, AResult, newSize);
    strassen(AResult, B22, M5, newSize);

    subtract(A21, A11, AResult, newSize);
    add(B11, B12, BResult, newSize);
    strassen(AResult, BResult, M6, newSize);

    subtract(A12, A22, AResult, newSize);
    add(B21, B22, BResult, newSize);
    strassen(AResult, BResult, M7, newSize);

    for (int i = 0; i < newSize; ++i)
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + newSize] = M3[i][j] + M5[i][j];
            C[i + newSize][j] = M2[i][j] + M4[i][j];
            C[i + newSize][j + newSize] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }

    freeMatrix(A11, newSize); freeMatrix(A12, newSize);
    freeMatrix(A21, newSize); freeMatrix(A22, newSize);
    freeMatrix(B11, newSize); freeMatrix(B12, newSize);
    freeMatrix(B21, newSize); freeMatrix(B22, newSize);
    freeMatrix(M1, newSize); freeMatrix(M2, newSize); freeMatrix(M3, newSize);
    freeMatrix(M4, newSize); freeMatrix(M5, newSize); freeMatrix(M6, newSize); freeMatrix(M7, newSize);
    freeMatrix(AResult, newSize); freeMatrix(BResult, newSize);
}

int** multiply(int** mat1, int mat1Size, int* mat1ColSize, int** mat2, int mat2Size, int* mat2ColSize, int* returnSize, int** returnColumnSizes) {
    int m = mat1Size;
    int k = mat1ColSize[0];
    int n = mat2ColSize[0];

    int N = nextPowerOfTwo(m > k ? (m > n ? m : n) : (k > n ? k : n));
    int** A = allocateMatrix(N);
    int** B = allocateMatrix(N);
    int** C = allocateMatrix(N);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < k; ++j)
            A[i][j] = mat1[i][j];

    for (int i = 0; i < k; ++i)
        for (int j = 0; j < n; ++j)
            B[i][j] = mat2[i][j];

    strassen(A, B, C, N);

    int** result = (int**)malloc(m * sizeof(int*));
    *returnColumnSizes = (int*)malloc(m * sizeof(int));
    for (int i = 0; i < m; ++i) {
        result[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; ++j)
            result[i][j] = C[i][j];
        (*returnColumnSizes)[i] = n;
    }
    *returnSize = m;

    freeMatrix(A, N);
    freeMatrix(B, N);
    freeMatrix(C, N);
    return result;
}
