#include <stdio.h>
#include <stdlib.h>

int cmp(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int canFinish(int* hens, int hensSize, int* grains, int grainsSize, int t) {
    int r = 0;
    for (int i = 0; i < hensSize; i++) {
        int hen = hens[i];
        if (hen > grains[r]) {
            if (hen - grains[r] > t) return 0;
            int diff = hen - grains[r];
            int budget = (t - diff) / 2;
            if (budget < t - 2 * diff) budget = t - 2 * diff;
            while (r < grainsSize && grains[r] - hen <= budget) r++;
        } else {
            int budget = t;
            while (r < grainsSize && grains[r] - hen <= budget) r++;
        }
        if (r == grainsSize) return 1;
    }
    return 0;
}

int minimumTime(int* hens, int hensSize, int* grains, int grainsSize) {
    qsort(hens, hensSize, sizeof(int), cmp);
    qsort(grains, grainsSize, sizeof(int), cmp);
    int lo = 0, hi = 2000000001;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (canFinish(hens, hensSize, grains, grainsSize, mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}
