#include <stdio.h>

int waysToDistribute(int n, int k) {
    const int MOD = 1000000007;
    static long long dp[1001][1001];
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j <= k; ++j)
            dp[i][j] = 0;
    for (int i = 0; i <= k; ++i)
        dp[i][i] = 1;
    for (int i = 1; i <= n; ++i)
        for (int j = 1; j <= k; ++j)
            dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j] * j) % MOD;
    return (int)dp[n][k];
}
