#include <stdio.h>
#include <stdlib.h>
#include <float.h>

double max(double a, double b) {
    return a > b ? a : b;
}

int is_concave(int x1, int x2, int x3, double* prefix) {
    int dx1 = x2 - x1;
    double dy1 = prefix[x2] - prefix[x1];
    int dx2 = x3 - x2;
    double dy2 = prefix[x3] - prefix[x2];
    return dx1 * dy2 < dy1 * dx2;
}

double slope(int i, int j, double* prefix) {
    return (prefix[j] - prefix[i]) / (j - i);
}

double findMaxAverage(int* nums, int numsSize, int k) {
    int n = numsSize;
    double* prefix = (double*)malloc((n + 1) * sizeof(double));
    prefix[0] = 0;
    for (int i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + nums[i];
    }

    int* dq = (int*)malloc((n + 1) * sizeof(int));
    int front = 0, back = 0;
    double res = -DBL_MAX;

    for (int j = k; j <= n; ++j) {
        int i = j - k;

        while (back - front >= 2 && is_concave(dq[back - 2], dq[back - 1], i, prefix)) {
            back--;
        }
        dq[back++] = i;

        while (back - front >= 2 && slope(dq[front], j, prefix) <= slope(dq[front + 1], j, prefix)) {
            front++;
        }

        res = max(res, slope(dq[front], j, prefix));
    }

    free(prefix);
    free(dq);
    return res;
}
