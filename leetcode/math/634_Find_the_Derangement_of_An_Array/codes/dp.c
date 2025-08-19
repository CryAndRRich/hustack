#include <stdio.h>
#include <stdlib.h>

#define MOD 1000000007

int findDerangement(int n) {
    if (n == 1) return 0;
    if (n == 2) return 1;

    long long* dp = (long long*)malloc((n + 1) * sizeof(long long));
    dp[1] = 0;
    dp[2] = 1;

    for (int i = 3; i <= n; i++) {
        dp[i] = ((i - 1LL) * (dp[i - 1] + dp[i - 2])) % MOD;
    }

    int result = dp[n];
    free(dp);
    return result;
}
