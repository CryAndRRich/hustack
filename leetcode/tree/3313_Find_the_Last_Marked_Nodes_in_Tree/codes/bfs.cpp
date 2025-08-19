#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> lastMarkedNodes(vector<vector<int>>& edges) {
        int n = edges.size() + 1;
        vector<vector<int>> g(n);
        for (auto &e : edges) {
            g[e[0]].push_back(e[1]);
            g[e[1]].push_back(e[0]);
        }

        auto bfs = [&](int start) {
            vector<int> dist(n, -1);
            dist[start] = 0;
            queue<int> q;
            q.push(start);
            int far = start;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                far = u;
                for (int v : g[u]) {
                    if (dist[v] == -1) {
                        dist[v] = dist[u] + 1;
                        q.push(v);
                    }
                }
            }
            return pair<int, vector<int>>(far, dist);
        };

        auto [f1, _] = bfs(0);
        auto [f2, dist1] = bfs(f1);
        auto [_, dist2] = bfs(f2);

        vector<int> res(n);
        for (int i = 0; i < n; i++) {
            res[i] = dist1[i] > dist2[i] ? f1 : f2;
        }
        return res;
    }
};
