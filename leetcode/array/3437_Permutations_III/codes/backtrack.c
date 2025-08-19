#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int** permute(int n, int* returnSize, int** returnColumnSizes) {
    int maxSize = 1;
    for (int i = 2; i <= n; i++) maxSize *= i;
    int** res = (int**)malloc(maxSize * sizeof(int*));
    *returnColumnSizes = (int*)malloc(maxSize * sizeof(int));
    bool* used = (bool*)calloc(n + 1, sizeof(bool));
    int* path = (int*)malloc(n * sizeof(int));
    *returnSize = 0;

    void backtrack(int len) {
        if (len == n) {
            res[*returnSize] = (int*)malloc(n * sizeof(int));
            for (int i = 0; i < n; i++) res[*returnSize][i] = path[i];
            (*returnColumnSizes)[*returnSize] = n;
            (*returnSize)++;
            return;
        }
        for (int num = 1; num <= n; num++) {
            if (!used[num]) {
                if (len == 0 || (path[len - 1] % 2 != num % 2)) {
                    used[num] = true;
                    path[len] = num;
                    backtrack(len + 1);
                    used[num] = false;
                }
            }
        }
    }

    backtrack(0);
    free(used);
    free(path);
    return res;
}
