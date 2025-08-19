#include <string.h>
int priceArr[105], tasteArr[105];
int dp[105][1005][7];
int n;
int dfs(int i, int budget, int coupons) {
    if (i == n) return 0;
    if (dp[i][budget][coupons] != -1) return dp[i][budget][coupons];
    int ans = dfs(i + 1, budget, coupons);
    if (budget - priceArr[i] >= 0) {
        int val = dfs(i + 1, budget - priceArr[i], coupons) + tasteArr[i];
        if (val > ans) ans = val;
    }
    if (coupons > 0 && budget - (priceArr[i] / 2) >= 0) {
        int val = dfs(i + 1, budget - (priceArr[i] / 2), coupons - 1) + tasteArr[i];
        if (val > ans) ans = val;
    }
    return dp[i][budget][coupons] = ans;
}
int maxTastiness(int* price, int priceSize, int* tastiness, int tastinessSize, int maxAmount, int maxCoupons) {
    n = priceSize;
    for (int i = 0; i < n; i++) {
        priceArr[i] = price[i];
        tasteArr[i] = tastiness[i];
    }
    memset(dp, -1, sizeof(dp));
    return dfs(0, maxAmount, maxCoupons);
}
