#include <bits/stdc++.h>
using namespace std;

int N, M;
vector<vector<int>> adj, adj_rev;
vector<bool> visited;
vector<int> order;

void dfs(int u) {
    visited[u] = true;
    for (int v : adj[u]) {
        if (!visited[v])
            dfs(v);
    }
    order.push_back(u);
}

void rev_dfs(int u) {
    visited[u] = true;
    for (int v : adj_rev[u]) {
        if (!visited[v])
            rev_dfs(v);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> N >> M;
    adj.assign(N, {});
    adj_rev.assign(N, {});
    visited.assign(N, false);
    order.reserve(N);

    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        adj[u - 1].push_back(v - 1);
        adj_rev[v - 1].push_back(u - 1);
    }

    for (int i = 0; i < N; i++) {
        if (!visited[i])
            dfs(i);
    }

    fill(visited.begin(), visited.end(), false);
    int scc_count = 0;
    for (int i = N - 1; i >= 0; i--) {
        int u = order[i];
        if (!visited[u]) {
            rev_dfs(u);
            ++scc_count;
        }
    }

    cout << scc_count;
    return 0;
}
