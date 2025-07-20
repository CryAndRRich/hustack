#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double probabilityOfHeads(double* prob, int probSize, int target) {
    double** dp = (double**)malloc((probSize + 1) * sizeof(double*));
    for (int i = 0; i <= probSize; ++i) {
        dp[i] = (double*)calloc(target + 1, sizeof(double));
    }

    dp[0][0] = 1.0;

    for (int i = 1; i <= probSize; ++i) {
        for (int k = 0; k <= target; ++k) {
            dp[i][k] = dp[i - 1][k] * (1 - prob[i - 1]);
            if (k > 0) {
                dp[i][k] += dp[i - 1][k - 1] * prob[i - 1];
            }
        }
    }

    double result = dp[probSize][target];

    for (int i = 0; i <= probSize; ++i) free(dp[i]);
    free(dp);

    return result;
}
