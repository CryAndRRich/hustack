#include <stdio.h>
#include <stdlib.h>

int numberOfWays(int numPeople) {
    int MOD = 1000000007;
    int n = numPeople / 2;
    long long* dp = (long long*)calloc(n + 1, sizeof(long long));
    dp[0] = 1;

    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < i; j++) {
            dp[i] = (dp[i] + dp[j] * dp[i - 1 - j]) % MOD;
        }
    }

    int res = dp[n];
    free(dp);
    return res;
}
