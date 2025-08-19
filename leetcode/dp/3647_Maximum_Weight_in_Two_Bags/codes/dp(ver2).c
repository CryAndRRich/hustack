#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

int maxWeight(int* weights, int weightsSize, int w1, int w2) {
    unsigned long long* dp = (unsigned long long*)calloc(w1 + 1, sizeof(unsigned long long));
    dp[0] = 1ULL;
    for (int idx = 0; idx < weightsSize; idx++) {
        int w = weights[idx];
        if (w > w1 && w > w2) continue;
        unsigned long long* new_dp = (unsigned long long*)malloc((w1 + 1) * sizeof(unsigned long long));
        memcpy(new_dp, dp, (w1 + 1) * sizeof(unsigned long long));
        if (w <= w1) {
            for (int i = w1; i >= w; i--) {
                new_dp[i] |= dp[i - w];
            }
        }
        if (w <= w2) {
            for (int i = 0; i <= w1; i++) {
                new_dp[i] |= (dp[i] << w);
            }
        }
        free(dp);
        dp = new_dp;
    }
    int ans = -1;
    for (int i = 0; i <= w1; i++) {
        for (int j = w2; j >= 0; j--) {
            if ((dp[i] >> j) & 1ULL) {
                if (i + j > ans) ans = i + j;
                break;
            }
        }
    }
    free(dp);
    return ans;
}
