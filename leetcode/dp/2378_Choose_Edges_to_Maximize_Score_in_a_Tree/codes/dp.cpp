#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long maxScore(vector<vector<int>>& edges) {
        int n = edges.size();
        vector<vector<pair<int,int>>> g(n);
        for (int i = 1; i < n; i++) {
            int p = edges[i][0], w = edges[i][1];
            g[p].push_back({i, w});
        }
        vector<array<long long,2>> dp(n, {LLONG_MIN, LLONG_MIN});

        function<long long(int,int)> dp = [&](int u, int used) -> long long {
            long long &res = dp[u][used];
            if (res != LLONG_MIN) return res;
            if (g[u].empty()) return res = 0;
            long long ans = 0;
            for (auto &e : g[u]) ans += dp(e.first, 0);

            if (!used) {
                for (auto &e : g[u]) {
                    int v = e.first, w = e.second;
                    if (w <= 0) continue;
                    long long tmp = 0;
                    for (auto &ee : g[u]) {
                        if (ee.first == v) tmp += w + dp(ee.first, 1);
                        else tmp += dp(ee.first, 0);
                    }
                    ans = max(ans, tmp);
                }
            }
            return res = ans;
        };

        return dp(0, 0);
    }
};
