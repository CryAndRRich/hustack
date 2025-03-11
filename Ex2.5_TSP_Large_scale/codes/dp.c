#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INF 1000000000000000000LL

int n;
int **dist;
long long **dp;
int **pre;

void solveTSP() {
    int fullMask = (1 << n) - 1;
    dp = (long long**)malloc(sizeof(long long*) * (1 << n));
    pre = (int**)malloc(sizeof(int*) * (1 << n));

    for (int mask = 0; mask < (1 << n); mask++) {
        dp[mask] = (long long*)malloc(sizeof(long long) * n);
        pre[mask] = (int*)malloc(sizeof(int) * n);
        for (int i = 0; i < n; i++) {
            dp[mask][i] = INF;
            pre[mask][i] = -1;
        }
    }

    for (int i = 0; i < n; i++) {
        dp[1 << i][i] = dist[i][i];
    }

    for (int mask = 0; mask < (1 << n); mask++) {
        for (int i = 0; i < n; i++) {
            if (!(mask & (1 << i))) continue;
            for (int j = 0; j < n; j++) {
                if (mask & (1 << j)) continue;
                int newMask = mask | (1 << j);
                long long newCost = dp[mask][i] + dist[i][j];
                if (newCost < dp[newMask][j]) {
                    dp[newMask][j] = newCost;
                    pre[newMask][j] = i;
                }
            }
        }
    }

    long long bestCost = INF;
    int lastCity = -1;
    for (int i = 0; i < n; i++) {
        long long cost = dp[fullMask][i] + dist[i][0];
        if (cost < bestCost) {
            bestCost = cost;
            lastCity = i;
        }
    }

    int *path = (int*)malloc(sizeof(int) * n);
    int mask = fullMask;
    int index = n - 1;
    while (lastCity != -1) {
        path[index--] = lastCity + 1; 
        int temp = pre[mask][lastCity];
        mask ^= (1 << lastCity);
        lastCity = temp;
    }

    printf("%d\n", n);
    for (int i = 0; i < n; i++) {
        printf("%d ", path[i]);
    }

    free(path);
    for (int i = 0; i < (1 << n); i++) {
        free(dp[i]);
        free(pre[i]);
    }
    free(dp);
    free(pre);
}

int main() {
    scanf("%d", &n);

    dist = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++) {
        dist[i] = (int*)malloc(sizeof(int) * n);
        for (int j = 0; j < n; j++) {
            scanf("%d", &dist[i][j]);
        }
    }

    solveTSP();

    for (int i = 0; i < n; i++) {
        free(dist[i]);
    }
    free(dist);

    return 0;
}
