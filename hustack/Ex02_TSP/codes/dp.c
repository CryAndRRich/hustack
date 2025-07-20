#include <stdio.h>
#include <limits.h>

#define MAX 20
#define INF 1000000000

int n;
int cost[MAX][MAX];
int dp[1 << MAX][MAX];

int main() {
    scanf("%d", &n);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &cost[i][j]);
        }
    }
    
    int fullMask = 1 << n;
    
    for (int mask = 0; mask < fullMask; mask++) {
        for (int i = 0; i < n; i++) {
            dp[mask][i] = INF;
        }
    }
    
    dp[1 << 0][0] = 0;
    
    for (int mask = 0; mask < fullMask; mask++) {
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                for (int j = 0; j < n; j++) {
                    if (!(mask & (1 << j))) {
                        int newMask = mask | (1 << j);
                        if (dp[mask][i] + cost[i][j] < dp[newMask][j]) {
                            dp[newMask][j] = dp[mask][i] + cost[i][j];
                        }
                    }
                }
            }
        }
    }
    int best = INF;
    for (int i = 1; i < n; i++) {
        if (dp[fullMask - 1][i] + cost[i][0] < best) {
            best = dp[fullMask - 1][i] + cost[i][0];
        }
    }
    
    printf("%d\n", best);
    return 0;
}