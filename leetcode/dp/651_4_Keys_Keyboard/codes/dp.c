int maxA(int n) {
    int dp[51] = {0};
    for (int i = 1; i <= n; i++) {
        dp[i] = dp[i - 1] + 1;
        for (int j = 2; j < i - 1; j++) {
            int curr = dp[j] * (i - j - 1);
            if (curr > dp[i]) {
                dp[i] = curr;
            }
        }
    }
    return dp[n];
}
