#include <stdio.h>
#include <stdlib.h>

int* findBuildings(int* heights, int heightsSize, int* returnSize) {
    int* result = (int*)malloc(sizeof(int) * heightsSize);
    int count = 0;
    int maxHeight = 0;

    for (int i = heightsSize - 1; i >= 0; --i) {
        if (count == 0 || heights[i] > maxHeight) {
            result[count++] = i;
            maxHeight = heights[i];
        }
    }

    for (int i = 0; i < count / 2; ++i) {
        int tmp = result[i];
        result[i] = result[count - 1 - i];
        result[count - 1 - i] = tmp;
    }

    *returnSize = count;
    return result;
}
