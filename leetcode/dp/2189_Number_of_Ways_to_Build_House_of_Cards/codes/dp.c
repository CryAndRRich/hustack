#include <string.h>

static int memo[501][168];

int dp(int rem, int cap) {
    if (rem == 0) return 1;
    if (cap <= 0) return 0;
    if (memo[rem][cap] != -1) return memo[rem][cap];
    int limit = cap;
    int t = (rem + 1) / 3;
    if (limit > t) limit = t;
    if (limit == 0) return memo[rem][cap] = 0;
    int ans = 0;
    for (int k = 1; k <= limit; k++) {
        ans += dp(rem - (3 * k - 1), k - 1);
    }
    return memo[rem][cap] = ans;
}

int houseOfCards(int n) {
    int maxK = (n + 1) / 3;
    for (int i = 0; i <= n; i++)
        for (int j = 0; j <= maxK; j++)
            memo[i][j] = -1;
    return dp(n, maxK);
}
