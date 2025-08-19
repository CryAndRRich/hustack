#include <stdlib.h>

int upper_bound(int* arr, int size, int target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

int minOperations(int* nums, int numsSize) {
    int* tails = (int*)malloc(sizeof(int) * numsSize);
    int size = 0;
    for (int i = 0; i < numsSize; ++i) {
        int neg = -nums[i];
        int idx = upper_bound(tails, size, neg);
        if (idx == size) tails[size++] = neg;
        else tails[idx] = neg;
    }
    free(tails);
    return size;
}
