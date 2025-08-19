#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int n;
    vector<int> p;
    vector<long long> memo;

    long long dfs(int mask) {
        if (mask == (1 << n) - 1) return 0;
        long long &res = memo[mask];
        if (res != -1) return res;
        int g = 1 + __builtin_popcount((unsigned)mask);
        long long best = LLONG_MAX;
        for (int i = 0; i < n; i++) {
            if (!((mask >> i) & 1)) {
                long long days = (p[i] + (long long)g - 1) / g;
                long long v = days + dfs(mask | (1 << i));
                if (v < best) best = v;
            }
        }
        return res = best;
    }

    long long minimumTime(vector<int>& power) {
        p = power;
        n = (int)p.size();
        memo.assign(1 << n, -1);
        return dfs(0);
    }
};
