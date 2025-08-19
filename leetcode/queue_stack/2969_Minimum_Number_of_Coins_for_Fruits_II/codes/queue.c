#include <stdio.h>
#include <stdlib.h>

#define MAXN 100005

int minimumCoins(int* prices, int n) {
    int dp[MAXN];
    int q[MAXN]; 
    int front = 0, back = 0;

    dp[0] = prices[0];
    q[back++] = 0;

    for (int i = 1; i < n; i++) {
        dp[i] = dp[q[front]] + prices[i];
        while (front < back && q[front] * 2 + 1 < i) {
            front++;
        }
        while (front < back && dp[q[back - 1]] >= dp[i]) {
            back--;
        }
        q[back++] = i;
    }
    return dp[q[front]];
}