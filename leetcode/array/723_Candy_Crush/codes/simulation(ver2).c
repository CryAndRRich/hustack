/**
 * Return an array of arrays of size *returnSize.
 * The sizes of the arrays are returned as *returnColumnSizes array.
 * Note: Both returned array and *columnSizes array must be malloced, assume caller calls free().
 */
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

int** rotate90(int** board, int m, int n) {
    int** res = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; ++i) {
        res[i] = (int*)malloc(sizeof(int) * m);
        for (int j = 0; j < m; ++j) {
            res[i][j] = board[m - 1 - j][i];
        }
    }
    return res;
}

int** rotateMinus90(int** board, int m, int n) {
    int** res = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; ++i) {
        res[i] = (int*)malloc(sizeof(int) * m);
        for (int j = 0; j < m; ++j) {
            res[i][j] = board[j][n - 1 - i];
        }
    }
    return res;
}

int** candyCrush(int** board, int boardSize, int* boardColSize, int* returnSize, int** returnColumnSizes) {
    int m = boardSize;
    int n = boardColSize[0];

    int** b = rotate90(board, m, n);
    int rows = n, cols = m;

    bool changed = true;
    while (changed) {
        changed = false;

        int** mark = (int**)malloc(sizeof(int*) * rows);
        for (int i = 0; i < rows; ++i) {
            mark[i] = (int*)calloc(cols, sizeof(int));
        }

        for (int i = 2; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int v = b[i][j];
                if (v != 0 && v == b[i - 1][j] && v == b[i - 2][j]) {
                    mark[i][j] = mark[i - 1][j] = mark[i - 2][j] = 1;
                    changed = true;
                }
            }
        }

        for (int i = 0; i < rows; ++i) {
            for (int j = 2; j < cols; ++j) {
                int v = b[i][j];
                if (v != 0 && v == b[i][j - 1] && v == b[i][j - 2]) {
                    mark[i][j] = mark[i][j - 1] = mark[i][j - 2] = 1;
                    changed = true;
                }
            }
        }

        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                if (mark[i][j]) b[i][j] = 0;

        for (int i = 0; i < rows; ++i) {
            int write = 0;
            for (int j = 0; j < cols; ++j) {
                if (b[i][j] != 0)
                    b[i][write++] = b[i][j];
            }
            while (write < cols)
                b[i][write++] = 0;
        }

        for (int i = 0; i < rows; ++i) free(mark[i]);
        free(mark);
    }

    int** result = rotateMinus90(b, rows, cols);

    for (int i = 0; i < rows; ++i) free(b[i]);
    free(b);

    *returnSize = m;
    *returnColumnSizes = (int*)malloc(sizeof(int) * m);
    for (int i = 0; i < m; ++i) {
        (*returnColumnSizes)[i] = n;
    }

    return result;
}
