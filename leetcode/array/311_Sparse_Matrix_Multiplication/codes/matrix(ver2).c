/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int col;
    int val;
} Pair;

int** multiply(int** mat1, int mat1Size, int* mat1ColSize, int** mat2, int mat2Size, int* mat2ColSize, int* returnSize, int** returnColumnSizes) {
    int m = mat1Size;
    int k = mat1ColSize[0];
    int n = mat2ColSize[0];

    int** result = (int**)malloc(m * sizeof(int*));
    *returnColumnSizes = (int*)malloc(m * sizeof(int));
    for (int i = 0; i < m; ++i) {
        result[i] = (int*)calloc(n, sizeof(int));
        (*returnColumnSizes)[i] = n;
    }
    *returnSize = m;

    Pair** mat1_map = (Pair**)malloc(m * sizeof(Pair*));
    int* mat1_mapSize = (int*)calloc(m, sizeof(int));
    for (int i = 0; i < m; ++i) {
        mat1_map[i] = (Pair*)malloc(k * sizeof(Pair));
        for (int j = 0; j < k; ++j) {
            if (mat1[i][j] != 0) {
                mat1_map[i][mat1_mapSize[i]].col = j;
                mat1_map[i][mat1_mapSize[i]].val = mat1[i][j];
                mat1_mapSize[i]++;
            }
        }
    }

    Pair** mat2_map = (Pair**)malloc(k * sizeof(Pair*));
    int* mat2_mapSize = (int*)calloc(k, sizeof(int));
    for (int i = 0; i < k; ++i) {
        mat2_map[i] = (Pair*)malloc(n * sizeof(Pair));
        for (int j = 0; j < n; ++j) {
            if (mat2[i][j] != 0) {
                mat2_map[i][mat2_mapSize[i]].col = j;
                mat2_map[i][mat2_mapSize[i]].val = mat2[i][j];
                mat2_mapSize[i]++;
            }
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int a = 0; a < mat1_mapSize[i]; ++a) {
            int t = mat1_map[i][a].col;
            int val1 = mat1_map[i][a].val;
            for (int b = 0; b < mat2_mapSize[t]; ++b) {
                int j = mat2_map[t][b].col;
                int val2 = mat2_map[t][b].val;
                result[i][j] += val1 * val2;
            }
        }
    }

    for (int i = 0; i < m; ++i) free(mat1_map[i]);
    for (int i = 0; i < k; ++i) free(mat2_map[i]);
    free(mat1_map); free(mat1_mapSize);
    free(mat2_map); free(mat2_mapSize);

    return result;
}
