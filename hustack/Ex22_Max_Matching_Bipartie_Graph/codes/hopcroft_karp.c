#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#define MAXN 10001
#define MAXM 10001

int n, m;
int adj[MAXN][MAXM];
int deg[MAXN];
int pairU[MAXN];
int pairV[MAXM];
int dist[MAXN];
int queue[MAXN];
int front, back;

int bfs() {
    front = back = 0;
    for (int u = 1; u <= n; ++u) {
        if (pairU[u] == 0) {
            dist[u] = 0;
            queue[back++] = u;
        } else {
            dist[u] = INT_MAX;
        }
    }

    int found = 0;
    while (front < back) {
        int u = queue[front++];
        for (int i = 0; i < deg[u]; ++i) {
            int v = adj[u][i];
            if (pairV[v] == 0) {
                found = 1;
            } else if (dist[pairV[v]] == INT_MAX) {
                dist[pairV[v]] = dist[u] + 1;
                queue[back++] = pairV[v];
            }
        }
    }

    return found;
}

int dfs(int u) {
    for (int i = 0; i < deg[u]; ++i) {
        int v = adj[u][i];
        if (pairV[v] == 0 || (dist[pairV[v]] == dist[u] + 1 && dfs(pairV[v]))) {
            pairU[u] = v;
            pairV[v] = u;
            return 1;
        }
    }
    dist[u] = INT_MAX;
    return 0;
}

int hopcroft_karp() {
    memset(pairU, 0, sizeof(pairU));
    memset(pairV, 0, sizeof(pairV));
    int matching = 0;
    while (bfs()) {
        for (int u = 1; u <= n; ++u) {
            if (pairU[u] == 0 && dfs(u)) {
                matching++;
            }
        }
    }
    return matching;
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

    printf("%d", hopcroft_karp());
    return 0;
}
