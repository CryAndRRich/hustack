#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INF 1000000000000000000LL

int findMinimumTime(int* strength, int strengthSize) {
    int n = strengthSize;

    long long** cost = (long long**)malloc(n * sizeof(long long*));
    for (int i = 0; i < n; i++) {
        cost[i] = (long long*)malloc(n * sizeof(long long));
        for (int j = 0; j < n; j++) {
            cost[i][j] = (strength[j] + i) / (i + 1);
        }
    }

    long long* u = (long long*)calloc(n + 1, sizeof(long long));
    long long* v = (long long*)calloc(n + 1, sizeof(long long));
    int* p = (int*)calloc(n + 1, sizeof(int));
    int* way = (int*)calloc(n + 1, sizeof(int));

    for (int i = 1; i <= n; i++) {
        p[0] = i;
        long long* minv = (long long*)malloc((n + 1) * sizeof(long long));
        char* used = (char*)calloc(n + 1, sizeof(char));
        for (int j = 0; j <= n; j++) {
            minv[j] = INF;
            used[j] = 0;
        }

        int j0 = 0;
        while (1) {
            used[j0] = 1;
            int i0 = p[j0];
            long long delta = INF;
            int j1 = -1;
            for (int j = 1; j <= n; j++) {
                if (!used[j]) {
                    long long cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if (p[j0] == 0) break;
        }

        while (1) {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if (j0 == 0) break;
        }

        free(minv);
        free(used);
    }

    long long total_cost = -v[0];

    for (int i = 0; i < n; i++) free(cost[i]);
    free(cost);
    free(u);
    free(v);
    free(p);
    free(way);

    return (int)total_cost;
}