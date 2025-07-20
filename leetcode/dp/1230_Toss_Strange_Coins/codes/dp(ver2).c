#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double probabilityOfHeads(double* prob, int probSize, int target) {
    double* dp = (double*)calloc(target + 1, sizeof(double));
    dp[0] = 1.0;

    for (int i = 0; i < probSize; ++i) {
        for (int k = target; k >= 0; --k) {
            dp[k] = dp[k] * (1 - prob[i]) + (k > 0 ? dp[k - 1] * prob[i] : 0);
        }
    }

    double result = dp[target];
    free(dp);
    return result;
}
