#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

int maxWeight(int* weights, int weightsSize, int w1, int w2) {
    int** dp = (int**)malloc((w1 + 1) * sizeof(int*));
    for (int i = 0; i <= w1; i++) {
        dp[i] = (int*)malloc((w2 + 1) * sizeof(int));
        for (int j = 0; j <= w2; j++) dp[i][j] = -1;
    }
    dp[0][0] = 0;
    for (int idx = 0; idx < weightsSize; idx++) {
        int w = weights[idx];
        int** new_dp = (int**)malloc((w1 + 1) * sizeof(int*));
        for (int i = 0; i <= w1; i++) {
            new_dp[i] = (int*)malloc((w2 + 1) * sizeof(int));
            for (int j = 0; j <= w2; j++) new_dp[i][j] = dp[i][j];
        }
        for (int i = 0; i <= w1; i++) {
            for (int j = 0; j <= w2; j++) {
                if (dp[i][j] == -1) continue;
                if (i + w <= w1)
                    if (new_dp[i + w][j] < dp[i][j] + w)
                        new_dp[i + w][j] = dp[i][j] + w;
                if (j + w <= w2)
                    if (new_dp[i][j + w] < dp[i][j] + w)
                        new_dp[i][j + w] = dp[i][j] + w;
            }
        }
        for (int i = 0; i <= w1; i++) {
            memcpy(dp[i], new_dp[i], (w2 + 1) * sizeof(int));
            free(new_dp[i]);
        }
        free(new_dp);
    }
    int ans = 0;
    for (int i = 0; i <= w1; i++)
        for (int j = 0; j <= w2; j++)
            if (dp[i][j] > ans) ans = dp[i][j];
    for (int i = 0; i <= w1; i++) free(dp[i]);
    free(dp);
    return ans;
}
