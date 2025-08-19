#include <stdio.h>
#include <limits.h>

int minimumLines(int** points, int pointsSize, int* pointsColSize) {
    int n = pointsSize;
    int maskLine[10][10];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                maskLine[i][j] = 0;
                continue;
            }
            int mask = 0;
            int x1 = points[i][0], y1 = points[i][1];
            int x2 = points[j][0], y2 = points[j][1];
            for (int k = 0; k < n; k++) {
                int x3 = points[k][0], y3 = points[k][1];
                if ((long long)(x2 - x1) * (y3 - y1) == (long long)(y2 - y1) * (x3 - x1)) {
                    mask |= (1 << k);
                }
            }
            maskLine[i][j] = mask;
        }
    }
    int fullMask = (1 << n) - 1;
    int dp[1 << 10];
    for (int i = 0; i < (1 << n); i++) dp[i] = n;
    dp[0] = 0;
    for (int mask = 0; mask < (1 << n); mask++) {
        if (dp[mask] >= n) continue;
        int first = -1;
        for (int i = 0; i < n; i++) {
            if (!(mask & (1 << i))) {
                first = i;
                break;
            }
        }
        if (first == -1) continue;
        int m1 = mask | (1 << first);
        if (dp[m1] > dp[mask] + 1) dp[m1] = dp[mask] + 1;
        for (int j = 0; j < n; j++) {
            if (j != first && !(mask & (1 << j))) {
                int newMask = mask | maskLine[first][j];
                if (dp[newMask] > dp[mask] + 1) dp[newMask] = dp[mask] + 1;
            }
        }
    }
    return dp[fullMask];
}
