#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int shortestPathWithHops(int n, vector<vector<int>>& edges, int s, int d, int k) {
        vector<vector<pair<int,int>>> g(n);
        for (auto &e : edges) {
            g[e[0]].push_back({e[1], e[2]});
            g[e[1]].push_back({e[0], e[2]});
        }
        vector<vector<int>> dist(n, vector<int>(k + 1, INT_MAX));
        dist[s][0] = 0;
        priority_queue<array<int,3>, vector<array<int,3>>, greater<>> pq;
        pq.push({0, s, 0});
        while (!pq.empty()) {
            auto [t, u, h] = pq.top();
            pq.pop();
            if (u == d) return t;
            if (t > dist[u][h]) continue;
            for (auto [v, w] : g[u]) {
                if (t + w < dist[v][h]) {
                    dist[v][h] = t + w;
                    pq.push({dist[v][h], v, h});
                }
                if (h < k && t < dist[v][h + 1]) {
                    dist[v][h + 1] = t;
                    pq.push({t, v, h + 1});
                }
            }
        }
        return -1;
    }
};
