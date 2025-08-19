#include <stdio.h>
#include <stdlib.h>

int cmp(const void* a, const void* b) {
    return (*(int*)b - *(int*)a); 
}

int check(int* timeReq, int n, long long maxTime, int splitTime) {
    int currentWBC = 1;
    long long curTime = 0;
    for (int i = 0; i < n; ++i) {
        if (currentWBC == 0) return 0;
        while (curTime + splitTime + timeReq[i] <= maxTime && currentWBC < n - i) {
            currentWBC <<= 1;
            curTime += splitTime;
        }
        currentWBC--;
    }
    return 1;
}

long long minEliminationTime(int* timeReq, int n, int splitTime) {
    qsort(timeReq, n, sizeof(int), cmp);
    long long left = timeReq[0], right = (long long)timeReq[0] + 20LL * splitTime;
    while (left < right) {
        long long mid = (left + right) / 2;
        if (check(timeReq, n, mid, splitTime)) right = mid;
        else left = mid + 1;
    }
    return left;
}
