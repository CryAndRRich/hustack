#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    int dp[101][1001][6];
    vector<int> price, taste;
    int n;
    int dfs(int i, int budget, int coupons) {
        if (i == n) return 0;
        if (dp[i][budget][coupons] != -1) return dp[i][budget][coupons];
        int ans = dfs(i + 1, budget, coupons);
        if (budget - price[i] >= 0)
            ans = max(ans, dfs(i + 1, budget - price[i], coupons) + taste[i]);
        if (coupons > 0 && budget - (price[i] / 2) >= 0)
            ans = max(ans, dfs(i + 1, budget - (price[i] / 2), coupons - 1) + taste[i]);
        return dp[i][budget][coupons] = ans;
    }
    int maxTastiness(vector<int>& price, vector<int>& tastiness, int maxAmount, int maxCoupons) {
        memset(dp, -1, sizeof(dp));
        this -> price = price;
        this -> taste = tastiness;
        n = price.size();
        return dfs(0, maxAmount, maxCoupons);
    }
};
