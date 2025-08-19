#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    vector<long long> minCost(int n, vector<vector<int>>& roads, vector<int>& appleCost, int k) {
        vector<vector<pair<int,int>>> g(n);
        for (auto &r : roads) {
            g[r[0]-1].push_back({r[1]-1, r[2]});
            g[r[1]-1].push_back({r[0]-1, r[2]});
        }
        vector<long long> ans(n);
        for (int s = 0; s < n; s++) {
            vector<long long> dist(n, LLONG_MAX);
            dist[s] = 0;
            priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
            pq.push({0, s});
            long long res = LLONG_MAX;
            while (!pq.empty()) {
                auto [d, u] = pq.top(); pq.pop();
                if (d > dist[u]) continue;
                res = min(res, (long long)appleCost[u] + (long long)(k+1)*d);
                for (auto &[v, w] : g[u]) {
                    long long nd = d + w;
                    if (nd < dist[v]) {
                        dist[v] = nd;
                        pq.push({nd, v});
                    }
                }
            }
            ans[s] = res;
        }
        return ans;
    }
};
