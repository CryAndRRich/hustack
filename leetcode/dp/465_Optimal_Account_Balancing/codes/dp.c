#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

int minTransfers(int** transactions, int transactionsSize, int* transactionsColSize) {
    int balance[12] = {0}, i, j, k;
    for (i = 0; i < transactionsSize; i++) {
        int from = transactions[i][0], to = transactions[i][1], amt = transactions[i][2];
        balance[from] -= amt;
        balance[to] += amt;
    }

    int nonzero[12], n = 0;
    for (i = 0; i < 12; i++) {
        if (balance[i] != 0)
            nonzero[n++] = balance[i];
    }

    int total[1 << 12], dp[1 << 12];
    for (i = 0; i < (1 << n); i++) {
        total[i] = 0;
        for (j = 0; j < n; j++) {
            if ((i >> j) & 1)
                total[i] += nonzero[j];
        }
        dp[i] = INT_MAX;
    }
    dp[0] = 0;

    for (i = 1; i < (1 << n); i++) {
        if (total[i] != 0) continue;
        for (j = i; j; j = (j - 1) & i) {
            if (total[j] == 0 && dp[i ^ j] != INT_MAX) {
                int cnt = 0;
                for (k = 0; k < n; k++) {
                    if ((j >> k) & 1) cnt++;
                }
                dp[i] = fmin(dp[i], dp[i ^ j] + cnt - 1);
            }
        }
    }

    return dp[(1 << n) - 1];
}
