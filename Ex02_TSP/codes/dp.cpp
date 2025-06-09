#include <iostream>
#include <algorithm>
using namespace std;

#define MAX 20
#define INF 1000000000

int n;
int cost[MAX][MAX];
int dp[1 << MAX][MAX];

int main() {
    cin >> n;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> cost[i][j];
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
                        dp[newMask][j] = min(dp[newMask][j], dp[mask][i] + cost[i][j]);
                    }
                }
            }
        }
    }
    
    int best = INF;
    for (int i = 1; i < n; i++) {
        best = min(best, dp[fullMask - 1][i] + cost[i][0]);
    }
    
    cout << best;
    return 0;
}
