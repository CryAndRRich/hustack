#include <bits/stdc++.h>
using namespace std;

typedef struct { 
    int to; 
    long long cost; 
    int idx; 
} edge;

void dijkstra(int src, int n, const vector<vector<edge>>& adj, vector<long long>& dist, vector<pair<int, int>>& parent) {
    dist.assign(n + 1, INT_MAX);
    parent.assign(n + 1, {-1, -1});
    dist[src] = 0;
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    pq.push({0, src});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto &e : adj[u]) {
            if (e.cost < 0) continue;
            long long nd = d + e.cost;
            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                parent[e.to] = {u, e.idx};
                pq.push({nd, e.to});
            }
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    cin >> n >> m;
    vector<tuple<int, int, long long>> edges(m);
    vector<vector<edge>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        long long c;
        cin >> u >> v >> c;
        edges[i] = {u, v, c};
        adj[u].push_back({v, c, i});
    }

    vector<long long> dist1, dist2, potential(n + 1);
    vector<pair<int, int>> par1;
    dijkstra(1, n, adj, dist1, par1);
    if (dist1[n] == INT_MAX) {
        cout << "NOT_FEASIBLE";
        return 0;
    }

    for (int i = 1; i <= n; i++) potential[i] = dist1[i];
    vector<bool> used(m, false);
    for (int v = n; v != 1; v = par1[v].first)
        used[par1[v].second] = true;

    vector<vector<edge>> adj2(n + 1);
    for (int i = 0; i < m; i++) {
        auto [u, v, c] = edges[i];
        long long rc = c + potential[u] - potential[v];
        if (!used[i]) adj2[u].push_back({v, rc, i});
        else adj2[v].push_back({u, -rc, i});
    }

    dist2.assign(n + 1, INT_MAX);
    dist2[1] = 0;
    for (int it = 0; it < n-1; it++) {
        for (int u = 1; u <= n; u++) {
            if (dist2[u] == INT_MAX) continue;
            for (auto &e : adj2[u]) {
                if (dist2[u] + e.cost < dist2[e.to]) {
                    dist2[e.to] = dist2[u] + e.cost;
                }
            }
        }
    }
    if (dist2[n] == INT_MAX) {
        cout << "NOT_FEASIBLE";
        return 0;
    }

    long long ans = 2 * potential[n] + dist2[n];
    cout << ans;
    return 0;
}
