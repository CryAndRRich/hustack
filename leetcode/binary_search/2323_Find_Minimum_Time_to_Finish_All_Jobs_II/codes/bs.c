#include <stdio.h>
#include <stdlib.h>

int cmpAsc(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int canFinish(int* jobs, int* workers, int n, int days) {
    int i = 0, j = 0;
    while (i < n && j < n) {
        if ((long long)workers[j] * days >= jobs[i]) i++;
        j++;
    }
    return i == n;
}

int minimumTime(int* jobs, int jobsSize, int* workers, int workersSize) {
    qsort(jobs, jobsSize, sizeof(int), cmpAsc);
    qsort(workers, workersSize, sizeof(int), cmpAsc);

    int left = 1, right = 100000; 
    for (int i = 0; i < jobsSize; i++)
        if (jobs[i] > right) right = jobs[i];

    while (left < right) {
        int mid = (left + right) / 2;
        if (canFinish(jobs, workers, jobsSize, mid))
            right = mid;
        else
            left = mid + 1;
    }
    return left;
}
