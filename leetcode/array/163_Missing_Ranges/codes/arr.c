#include <stdlib.h>

int** findMissingRanges(int* nums, int numsSize, int lower, int upper, int* returnSize, int** returnColumnSizes) {
    int capacity = numsSize + 2;
    int** res = (int**)malloc(capacity * sizeof(int*));
    *returnColumnSizes = (int*)malloc(capacity * sizeof(int));
    *returnSize = 0;

    int prev = lower - 1;
    for (int i = 0; i <= numsSize; ++i) {
        int curr = (i == numsSize) ? upper + 1 : nums[i];
        if (curr - prev > 1) {
            res[*returnSize] = (int*)malloc(2 * sizeof(int));
            res[*returnSize][0] = prev + 1;
            res[*returnSize][1] = curr - 1;
            (*returnColumnSizes)[*returnSize] = 2;
            (*returnSize)++;
        }
        prev = curr;
    }
    return res;
}
