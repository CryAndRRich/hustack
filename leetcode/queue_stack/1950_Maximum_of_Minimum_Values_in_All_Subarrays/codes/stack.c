#include <stdio.h>
#include <stdlib.h>

int* findMaximums(int* nums, int numsSize, int* returnSize) {
    int* left = (int*)malloc(sizeof(int) * numsSize);
    int* right = (int*)malloc(sizeof(int) * numsSize);
    int* res = (int*)calloc(numsSize, sizeof(int));
    int* stack = (int*)malloc(sizeof(int) * numsSize);
    int top = -1;

    for (int i = 0; i < numsSize; ++i) left[i] = -1;
    for (int i = 0; i < numsSize; ++i) {
        while (top >= 0 && nums[stack[top]] >= nums[i]) --top;
        if (top >= 0) left[i] = stack[top];
        stack[++top] = i;
    }

    top = -1;
    for (int i = 0; i < numsSize; ++i) right[i] = numsSize;
    for (int i = numsSize - 1; i >= 0; --i) {
        while (top >= 0 && nums[stack[top]] >= nums[i]) --top;
        if (top >= 0) right[i] = stack[top];
        stack[++top] = i;
    }

    for (int i = 0; i < numsSize; ++i) {
        int len = right[i] - left[i] - 1;
        if (res[len - 1] < nums[i]) res[len - 1] = nums[i];
    }

    for (int i = numsSize - 2; i >= 0; --i)
        if (res[i] < res[i + 1]) res[i] = res[i + 1];

    *returnSize = numsSize;
    free(left); free(right); free(stack);
    return res;
}
