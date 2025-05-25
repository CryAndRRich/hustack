#include <stdio.h>
#include <string.h>

#define MAXN 10001
#define MAXM 10001

int n, m;
int adj[MAXN][MAXM];
int deg[MAXN];
int match[MAXM];
int visited[MAXN];

int dfs(int u) {
    if (visited[u]) return 0;
    visited[u] = 1;
    for (int i = 0; i < deg[u]; ++i) {
        int v = adj[u][i];
        if (match[v] == -1 || dfs(match[v])) {
            match[v] = u;
            return 1;
        }
    }
    return 0;
}

int main() {
    scanf("%d %d", &n, &m);
    for (int i = 1; i <= n; ++i) {
        int k;
        scanf("%d", &k);
        for (int j = 0; j < k; ++j) {
            int v;
            scanf("%d", &v);
            adj[i][deg[i]++] = v;
        }
    }

    memset(match, -1, sizeof(match));
    int res = 0;
    for (int u = 1; u <= n; ++u) {
        memset(visited, 0, sizeof(visited));
        res += dfs(u);
    }

    printf("%d", res);
    return 0;
}
