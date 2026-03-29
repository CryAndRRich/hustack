#include <stdio.h>
#include <stdbool.h>

#define MAX 20

int n, m;
int adj[MAX][MAX];
int path[MAX];
bool visited[MAX];

bool is_safe(int v, int pos) {
    if (adj[path[pos - 1]][v] == 0)
        return false;

    if (visited[v])
        return false;

    return true;
}

bool hamiltonian(int pos) {
    if (pos == n) {
        if (adj[path[pos - 1]][path[0]] == 1) {
            return true;
        } else {
            return false;
        }
    }

    for (int v = 1; v <= n; v++) {
        if (is_safe(v, pos)) {
            path[pos] = v;
            visited[v] = true;

            if (hamiltonian(pos + 1)) {
                return true;
            }

            visited[v] = false;
        }
    }

    return false;
}

void solve() {
    if (scanf("%d %d", &n, &m) != 2) {
        return;
    }

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            adj[i][j] = 0;
        }
        visited[i] = false;
    }

    for (int i = 0; i < m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        adj[u][v] = 1;
        adj[v][u] = 1;
    }

    path[0] = 1;
    visited[1] = true;

    if (hamiltonian(1))
        printf("1\n");
    else
        printf("0\n");
}

int main() {
    int t;
    if (scanf("%d", &t) != 1) {
        return 0;
    }
    
    while (t--) {
        solve();
    }
    
    return 0;
}