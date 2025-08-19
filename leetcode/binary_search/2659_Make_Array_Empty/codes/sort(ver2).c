#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int val;
    int idx;
} Pair;

int comparePairs(const void* a, const void* b) {
    return ((Pair*)a)->val - ((Pair*)b)->val;
}

long long countOperationsToEmptyArray(int* nums, int numsSize) {
    Pair* arr = malloc(sizeof(Pair) * numsSize);
    for (int i = 0; i < numsSize; i++) {
        arr[i].val = nums[i];
        arr[i].idx = i;
    }

    qsort(arr, numsSize, sizeof(Pair), comparePairs);

    long long operations = numsSize;
    for (int i = 1; i < numsSize; i++) {
        if (arr[i].idx < arr[i - 1].idx) {
            operations += numsSize - i;
        }
    }

    free(arr);
    return operations;
}