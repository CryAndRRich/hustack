#include <stdio.h>
#include <stdlib.h>

int cmp(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int maxLength(int* ribbons, int ribbonsSize, int k) {
    int maxVal = 0;
    for (int i = 0; i < ribbonsSize; i++)
        if (ribbons[i] > maxVal) maxVal = ribbons[i];
    int left = 1, right = maxVal, ans = 0;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        long long count = 0;
        for (int i = 0; i < ribbonsSize; i++)
            count += ribbons[i] / mid;
        if (count >= k) {
            ans = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return ans;
}
