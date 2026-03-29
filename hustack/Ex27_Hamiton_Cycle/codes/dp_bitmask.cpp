#include <iostream>
#include <cstring>

using namespace std;

bool dp[1 << 20][20];
int adj[20][20];

void solve() {
    int n, m;
    if (!(cin >> n >> m)) {
        return;
    }

    memset(adj, 0, sizeof(adj));
    
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        u--; 
        v--; 
        adj[u][v] = 1;
        adj[v][u] = 1;
    }

    int limit = 1 << n;

    for (int i = 0; i < limit; i++) {
        memset(dp[i], 0, n * sizeof(bool));
    }

    dp[1][0] = true;
    for (int mask = 1; mask < limit; mask++) {
        if ((mask & 1) == 0) {
            continue;
        }

        for (int u = 0; u < n; u++) {
            if (dp[mask][u]) {
                for (int v = 0; v < n; v++) {
                    if (adj[u][v] && !(mask & (1 << v))) {
                        dp[mask | (1 << v)][v] = true;
                    }
                }
            }
        }
    }

    bool has_cycle = false;
    int full_mask = limit - 1; 
    
    for (int u = 1; u < n; u++) {
        if (dp[full_mask][u] && adj[u][0]) {
            has_cycle = true;
            break;
        }
    }

    if (has_cycle) {
        cout << 1 << "\n";
    } else {
        cout << 0 << "\n";
    }
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int t;
    if (cin >> t) {
        while (t--) {
            solve();
        }
    }
    return 0;
}