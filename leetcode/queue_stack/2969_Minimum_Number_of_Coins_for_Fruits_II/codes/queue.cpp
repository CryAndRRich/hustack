#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int minimumCoins(vector<int>& prices) {
        int n = prices.size();
        vector<int> dp(n);
        dp[0] = prices[0];
        deque<int> q;
        q.push_back(0);

        for (int i = 1; i < n; i++) {
            dp[i] = dp[q.front()] + prices[i];
            while (!q.empty() && q.front() * 2 + 1 < i) {
                q.pop_front();
            }
            while (!q.empty() && dp[q.back()] >= dp[i]) {
                q.pop_back();
            }
            q.push_back(i);
        }
        return dp[q.front()];
    }
};
