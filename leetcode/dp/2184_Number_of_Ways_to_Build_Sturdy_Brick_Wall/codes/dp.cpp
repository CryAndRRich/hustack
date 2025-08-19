#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    static const int MOD = 1000000007;
    int buildWall(int height, int width, vector<int>& bricks) {
        vector<int> rows;
        function<void(int,int)> gen = [&](int pos, int mask) {
            if (pos == width) { rows.push_back(mask); return; }
            for (int b : bricks)
                if (pos + b <= width)
                    if (pos + b < width) gen(pos + b, mask | (1 << (pos + b)));
                    else gen(pos + b, mask);
        };
        gen(0,0);
        unordered_map<int, vector<int>> compat;
        for (int a : rows) compat[a] = {};
        for (int a : rows)
            for (int b : rows)
                if ((a & b) == 0) compat[a].push_back(b);
        unordered_map<int,long long> dp;
        for (int m : rows) dp[m] = 1;
        for (int i = 0; i < height - 1; i++) {
            unordered_map<int,long long> ndp;
            for (int m : rows) ndp[m] = 0;
            for (int m : rows)
                for (int nxt : compat[m])
                    ndp[nxt] = (ndp[nxt] + dp[m]) % MOD;
            dp = move(ndp);
        }
        long long ans = 0;
        for (auto &p : dp) ans = (ans + p.second) % MOD;
        return (int)ans;
    }
};
