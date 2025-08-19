#include <stdio.h>
#include <stdlib.h>

int cmpInt(const void* a, const void* b) {
    int x = *(int*)a, y = *(int*)b;
    return (x > y) - (x < y);
}

int binarySearchLower(int* arr, int size, int val) {
    int l = 0, r = size;
    while (l < r) {
        int m = (l + r) / 2;
        if (arr[m] < val) l = m + 1;
        else r = m;
    }
    return l;
}

int maximumTripletValue(int* nums, int n) {
    int* right = (int*)malloc(n * sizeof(int));
    int maxVal = 0;
    for (int i = n - 1; i >= 0; i--) {
        right[i] = maxVal;
        if (nums[i] > maxVal) maxVal = nums[i];
    }
    int* sorted = (int*)malloc(n * sizeof(int));
    int size = 0, ans = 0;
    for (int i = 0; i < n; i++) {
        if (size > 0 && right[i] > nums[i]) {
            int idx = binarySearchLower(sorted, size, nums[i]);
            if (idx > 0) {
                int greatestLeft = sorted[idx - 1];
                int val = greatestLeft - nums[i] + right[i];
                if (val > ans) ans = val;
            }
        }
        int pos = binarySearchLower(sorted, size, nums[i]);
        for (int j = size; j > pos; j--) sorted[j] = sorted[j - 1];
        sorted[pos] = nums[i];
        size++;
    }
    free(right);
    free(sorted);
    return ans;
}
