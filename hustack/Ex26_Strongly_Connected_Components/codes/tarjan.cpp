#include <bits/stdc++.h>
using namespace std;

int N, M;
vector<vector<int>> adj;
vector<int> dfn, low, stk;
vector<bool> in_stack;
int timer = 0, scc_count = 0;

void tarjan(int u) {
    dfn[u] = low[u] = ++timer;
    stk.push_back(u);
    in_stack[u] = true;

    for (int v : adj[u]) {
        if (dfn[v] == 0) {
            tarjan(v);
            low[u] = min(low[u], low[v]);
        }
        else if (in_stack[v]) {
            low[u] = min(low[u], dfn[v]);
        }
    }

    if (dfn[u] == low[u]) {
        while (true) {
            int v = stk.back(); 
            stk.pop_back();
            in_stack[v] = false;
            if (v == u) break;
        }
        ++scc_count;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> N >> M;
    adj.assign(N, {});
    dfn.assign(N, 0);
    low.assign(N, 0);
    in_stack.assign(N, false);
    stk.reserve(N);

    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        adj[u - 1].push_back(v - 1);
    }

    for (int i = 0; i < N; i++) {
        if (dfn[i] == 0)
            tarjan(i);
    }

    cout << scc_count;
    return 0;
}
