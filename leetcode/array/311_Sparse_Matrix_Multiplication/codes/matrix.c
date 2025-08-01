/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
#include <stdlib.h>

int** multiply(int** mat1, int mat1Size, int* mat1ColSize, int** mat2, int mat2Size, int* mat2ColSize, int* returnSize, int** returnColumnSizes) {
    int m = mat1Size;
    int k = *mat1ColSize;
    int n = *mat2ColSize;
    
    int** result = (int**)malloc(m * sizeof(int*));
    *returnColumnSizes = (int*)malloc(m * sizeof(int));
    
    for (int i = 0; i < m; ++i) {
        result[i] = (int*)calloc(n, sizeof(int));
        (*returnColumnSizes)[i] = n;
    }

    for (int i = 0; i < m; ++i) {
        for (int t = 0; t < k; ++t) {
            if (mat1[i][t] != 0) {
                for (int j = 0; j < n; ++j) {
                    if (mat2[t][j] != 0) {
                        result[i][j] += mat1[i][t] * mat2[t][j];
                    }
                }
            }
        }
    }

    *returnSize = m;
    return result;
}
