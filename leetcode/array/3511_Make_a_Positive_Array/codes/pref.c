#include <stdio.h>
#include <stdlib.h>

int makeArrayPositive(int* nums, int numsSize) {
    int n = numsSize;
    long long* pref = (long long*)malloc((n+1) * sizeof(long long));
    pref[0] = 0;
    for (int i = 0; i < n; ++i) pref[i+1] = pref[i] + (long long)nums[i];
    int res = 0;
    int covered = -1;
    for (int end = 2; end < n; ++end) {
        int need = 0;
        for (int l = 3; l <= 5; ++l) {
            int start = end - (l - 1);
            if (start < 0) break;
            if (covered >= start && covered <= end) continue;
            if (pref[end+1] - pref[start] <= 0) { need = 1; break; }
        }
        if (need) { res++; covered = end; }
    }
    free(pref);
    return res;
}
