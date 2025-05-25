#include <iostream>
#include <vector>
#include <queue>
#include <climits>
#include <algorithm>

using namespace std;

const int MAXN = 10001;
const int MAXM = 10001;

int n, m;
vector<int> adj[MAXN];    
int pairU[MAXN], pairV[MAXM];
int dist[MAXN];

bool bfs() {
    queue<int> q;
    for (int u = 1; u <= n; ++u) {
        if (pairU[u] == 0) {
            dist[u] = 0;
            q.push(u);
        } else {
            dist[u] = INT_MAX;
        }
    }

    bool found = false;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) {
            if (pairV[v] == 0) {
                found = true;  
            } else if (dist[pairV[v]] == INT_MAX) {
                dist[pairV[v]] = dist[u] + 1;
                q.push(pairV[v]);
            }
        }
    }

    return found;
}

bool dfs(int u) {
    for (int v : adj[u]) {
        if (pairV[v] == 0 || (dist[pairV[v]] == dist[u] + 1 && dfs(pairV[v]))) {
            pairU[u] = v;
            pairV[v] = u;
            return true;
        }
    }
    dist[u] = INT_MAX;
    return false;
}

int hopcroft_karp() {
    fill(pairU, pairU + MAXN, 0);
    fill(pairV, pairV + MAXM, 0);

    int matching = 0;
    while (bfs()) {
        for (int u = 1; u <= n; ++u) {
            if (pairU[u] == 0 && dfs(u)) {
                ++matching;
            }
        }
    }
    return matching;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n >> m;
    for (int u = 1; u <= n; ++u) {
        int k;
        cin >> k;
        while (k--) {
            int v;
            cin >> v;
            adj[u].push_back(v);
        }
    }

    cout << hopcroft_karp();
    return 0;
}
