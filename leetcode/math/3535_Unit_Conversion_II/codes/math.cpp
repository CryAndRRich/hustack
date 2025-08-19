#include <bits/stdc++.h>
using namespace std;
const long long MOD = 1000000007;

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e) {
        if (e & 1) r = (r * a) % MOD;
        a = (a * a) % MOD;
        e >>= 1;
    }
    return r;
}

class Solution {
public:
    vector<int> queryConversions(vector<vector<int>>& conversions, vector<vector<int>>& queries) {
        int n = conversions.size() + 1;
        vector<vector<array<long long,3>>> g(n);
        for (auto &c : conversions) {
            int a = c[0], b = c[1], f = c[2];
            g[a].push_back({b, f, 1});
            g[b].push_back({a, f, 0});
        }
        vector<long long> num(n), den(n);
        vector<int> vis(n, 0);
        num[0] = 1; den[0] = 1; vis[0] = 1;
        queue<int> q;
        q.push(0);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto &e : g[u]) {
                int v = e[0], f = e[1], forward = e[2];
                if (!vis[v]) {
                    if (forward) {
                        num[v] = num[u];
                        den[v] = (den[u] * f) % MOD;
                    } else {
                        num[v] = (num[u] * f) % MOD;
                        den[v] = den[u];
                    }
                    vis[v] = 1;
                    q.push(v);
                }
            }
        }
        vector<int> ans;
        for (auto &qq : queries) {
            int a = qq[0], b = qq[1];
            long long pa = num[a], qa = den[a];
            long long pb = num[b], qb = den[b];
            long long p = (pa * qb) % MOD;
            long long qv = (qa * pb) % MOD;
            long long inv_q = modpow(qv, MOD - 2);
            ans.push_back((p * inv_q) % MOD);
        }
        return ans;
    }
};
