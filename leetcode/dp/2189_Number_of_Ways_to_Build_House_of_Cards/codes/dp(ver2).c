#include <stdio.h>

int houseOfCards(int n) {
    static int dp[501];
    static int initialized = 0;
    if (!initialized) {
        dp[0] = 1;
        for (int row_size = 2; row_size <= 500; row_size += 3) {
            for (int reachable = 500; reachable >= row_size; reachable--) {
                dp[reachable] += dp[reachable - row_size];
            }
        }
        initialized = 1;
    }
    return dp[n];
}
