#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INIT_CAP 128
#define MAX_PATH 64

void ensure_capacity(int*** res, int** colSizes, int* cap, int required) {
    if (required >= *cap) {
        *cap *= 2;
        *res = realloc(*res, sizeof(int*) * (*cap));
        *colSizes = realloc(*colSizes, sizeof(int) * (*cap));
    }
}

void dfs(int n, int start, int* path, int pathLen, int*** res, int* returnSize, int** colSizes, int* cap) {
    for (int i = start; i <= sqrt(n); ++i) {
        if (n % i == 0) {
            path[pathLen] = i;
            path[pathLen + 1] = n / i;

            ensure_capacity(res, colSizes, cap, *returnSize);

            (*res)[*returnSize] = malloc(sizeof(int) * (pathLen + 2));
            for (int j = 0; j < pathLen + 2; ++j) {
                (*res)[*returnSize][j] = path[j];
            }

            (*colSizes)[*returnSize] = pathLen + 2;
            (*returnSize)++;

            dfs(n / i, i, path, pathLen + 1, res, returnSize, colSizes, cap);
        }
    }
}

int** getFactors(int n, int* returnSize, int** returnColumnSizes) {
    *returnSize = 0;
    int cap = INIT_CAP;
    int** res = malloc(sizeof(int*) * cap);
    *returnColumnSizes = malloc(sizeof(int) * cap);

    int* path = malloc(sizeof(int) * MAX_PATH);
    dfs(n, 2, path, 0, &res, returnSize, returnColumnSizes, &cap);
    free(path);
    return res;
}
