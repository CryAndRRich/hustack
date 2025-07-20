#include <bits/stdc++.h>
using namespace std;

class Solution {
    static const int LOG = 20;
    vector<vector<int>> adj, up;
    vector<int> depth;

    void dfs(int u, int p) {
        up[u][0] = p;
        for (int i = 1; i < LOG; ++i) {
            if (up[u][i-1] != -1)
                up[u][i] = up[up[u][i-1]][i-1];
        }
        for (int v : adj[u]) {
            if (v != p) {
                depth[v] = depth[u] + 1;
                dfs(v, u);
            }
        }
    }

    int lca(int u, int v) {
        if (depth[u] < depth[v]) swap(u, v);
        for (int i = LOG - 1; i >= 0; --i) {
            if (up[u][i] != -1 && depth[up[u][i]] >= depth[v])
                u = up[u][i];
        }
        if (u == v) return u;
        for (int i = LOG - 1; i >= 0; --i) {
            if (up[u][i] != -1 && up[u][i] != up[v][i]) {
                u = up[u][i];
                v = up[v][i];
            }
        }
        return up[u][0];
    }

    int dist(int u, int v) {
        int ancestor = lca(u, v);
        return depth[u] + depth[v] - 2 * depth[ancestor];
    }

    vector<int> getPathNodes(int u, int v) {
        vector<int> path, rev;
        int l = lca(u, v);
        while (u != l) {
            path.push_back(u);
            u = up[u][0];
        }
        path.push_back(l);
        while (v != l) {
            rev.push_back(v);
            v = up[v][0];
        }
        reverse(rev.begin(), rev.end());
        path.insert(path.end(), rev.begin(), rev.end());
        return path;
    }

public:
    vector<int> closestNode(int n, vector<vector<int>>& edges, vector<vector<int>>& query) {
        adj.assign(n, {});
        up.assign(n, vector<int>(LOG, -1));
        depth.assign(n, 0);

        for (auto& e : edges) {
            int u = e[0], v = e[1];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }

        dfs(0, -1);

        vector<int> ans;
        for (auto& q : query) {
            int start = q[0], end = q[1], node = q[2];
            vector<int> path = getPathNodes(start, end);
            int minDist = INT_MAX, res = -1;
            for (int x : path) {
                int d = dist(x, node);
                if (d < minDist) {
                    minDist = d;
                    res = x;
                }
            }
            ans.push_back(res);
        }
        return ans;
    }
};
