#include <stdlib.h>

static int cmp_interval(const void *pa, const void *pb) {
    int * const *a = pa;
    int * const *b = pb;
    int *ai = *a, *bi = *b;
    if (ai[0] < bi[0]) return -1;
    if (ai[0] > bi[0]) return 1;
    if (ai[1] < bi[1]) return -1;
    if (ai[1] > bi[1]) return 1;
    return 0;
}

int minConnectedGroups(int** intervals, int intervalsSize, int* intervalsColSize, int k) {
    if (intervalsSize == 0) return 0;
    qsort(intervals, intervalsSize, sizeof(int*), cmp_interval);
    long long *starts = (long long*)malloc(sizeof(long long) * intervalsSize);
    long long *ends = (long long*)malloc(sizeof(long long) * intervalsSize);
    int m = 0;
    for (int i = 0; i < intervalsSize; ++i) {
        long long s = intervals[i][0];
        long long e = intervals[i][1];
        if (m == 0 || s > ends[m-1]) {
            starts[m] = s;
            ends[m] = e;
            m++;
        } else {
            if (e > ends[m-1]) ends[m-1] = e;
        }
    }
    int res = m;
    for (int i = 0; i < m; ++i) {
        int lo = i, hi = m - 1, best = i;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (starts[mid] - ends[i] <= (long long)k) {
                best = mid;
                lo = mid + 1;
            } else hi = mid - 1;
        }
        int reduction = best - i;
        int candidate = m - reduction;
        if (candidate < res) res = candidate;
    }
    free(starts);
    free(ends);
    return res;
}
