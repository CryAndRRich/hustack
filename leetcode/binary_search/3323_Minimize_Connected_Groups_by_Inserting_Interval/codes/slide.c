#include <stdio.h>
#include <stdlib.h>

static int cmp_ptr(const void* a, const void* b) {
    int *pa = *(int**)a;
    int *pb = *(int**)b;
    if (pa[0] < pb[0]) return -1;
    if (pa[0] > pb[0]) return 1;
    if (pa[1] < pb[1]) return -1;
    if (pa[1] > pb[1]) return 1;
    return 0;
}

int minConnectedGroups(int** intervals, int intervalsSize, int* intervalsColSize, int k) {
    if (intervalsSize == 0) return 0;
    qsort(intervals, intervalsSize, sizeof(int*), cmp_ptr);
    long long (*merged)[2] = malloc(sizeof(long long) * 2 * intervalsSize);
    int mCount = 0;
    for (int i = 0; i < intervalsSize; ++i) {
        long long s = intervals[i][0];
        long long e = intervals[i][1];
        if (mCount == 0 || s > merged[mCount - 1][1]) {
            merged[mCount][0] = s;
            merged[mCount][1] = e;
            mCount++;
        } else {
            if (e > merged[mCount - 1][1]) merged[mCount - 1][1] = e;
        }
    }
    if (mCount == 0) { free(merged); return 0; }
    int j = 0;
    int maxLen = 1;
    for (int i = 0; i < mCount; ++i) {
        if (j < i) j = i;
        while (j + 1 < mCount && merged[j + 1][0] - merged[i][1] <= (long long)k) j++;
        int cur = j - i + 1;
        if (cur > maxLen) maxLen = cur;
    }
    free(merged);
    return mCount - maxLen + 1;
}
