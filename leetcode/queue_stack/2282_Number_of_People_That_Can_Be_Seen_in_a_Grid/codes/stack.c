#include <stdlib.h>
#include <stdbool.h>

#define MAX 410

int** seePeople(int** heights, int heightsSize, int* heightsColSize, int* returnSize, int** returnColumnSizes) {
    int r = heightsSize;
    int c = heightsColSize[0];

    int** res = (int**)malloc(r * sizeof(int*));
    *returnColumnSizes = (int*)malloc(r * sizeof(int));
    for (int i = 0; i < r; ++i) {
        res[i] = (int*)calloc(c, sizeof(int));
        (*returnColumnSizes)[i] = c;
    }

    int stack[MAX], top;

    for (int i = 0; i < r; ++i) {
        top = -1;
        for (int j = c - 1; j >= 0; --j) {
            bool equal = false;
            while (top >= 0 && stack[top] <= heights[i][j]) {
                if (stack[top] == heights[i][j]) equal = true;
                top--;
                res[i][j]++;
            }
            if (top >= 0 && !equal) res[i][j]++;
            stack[++top] = heights[i][j];
        }
    }

    for (int j = 0; j < c; ++j) {
        top = -1;
        for (int i = r - 1; i >= 0; --i) {
            bool equal = false;
            while (top >= 0 && stack[top] <= heights[i][j]) {
                if (stack[top] == heights[i][j]) equal = true;
                top--;
                res[i][j]++;
            }
            if (top >= 0 && !equal) res[i][j]++;
            stack[++top] = heights[i][j];
        }
    }

    *returnSize = r;
    return res;
}
