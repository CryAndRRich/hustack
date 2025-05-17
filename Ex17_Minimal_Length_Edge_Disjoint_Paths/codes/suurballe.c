#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MAXN 35
#define MAXM 1000

typedef struct { 
    int to; 
    long long cost; 
    int idx; 
} edge;

static inline void dijkstra(int src, int n, edge adj[][MAXM], int deg[], long long dist[], int parent[][2]) {
    int visited[MAXN] = {0};
    for (int i = 1; i <= n; i++) {
        dist[i] = INT_MAX;
        parent[i][0] = parent[i][1] = -1;
    }
    dist[src] = 0;
    for (int it = 0; it < n; it++) {
        int u = -1;
        long long best = INT_MAX;
        for (int i = 1; i <= n; i++) {
            if (!visited[i] && dist[i] < best) {
                best = dist[i]; u = i;
            }
        }
        if (u == -1) break;
        visited[u] = 1;
        for (int k = 0; k < deg[u]; k++) {
            edge e = adj[u][k];
            if (e.cost < 0) continue;
            long long nd = dist[u] + e.cost;
            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                parent[e.to][0] = u;
                parent[e.to][1] = e.idx;
            }
        }
    }
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    int u[MAXM], v[MAXM]; 
    long long c[MAXM];
    static edge adj[MAXN][MAXM], adj2[MAXN][MAXM];
    int deg[MAXN] = {0}, deg2[MAXN] = {0};
    for (int i = 0; i < m; i++) {
        scanf("%d %d %lld", &u[i], &v[i], &c[i]);
        adj[u[i]][deg[u[i]]++] = (edge){v[i], c[i], i, 0};
    }
    long long dist1[MAXN], dist2[MAXN], potential[MAXN];
    static int par1[MAXN][2], par2[MAXN][2];
    int used[MAXM] = {0};
    dijkstra(1, n, adj, deg, dist1, par1);
    
    if (dist1[n] >= INT_MAX) { 
        printf("NOT_FEASIBLE"); 
        return 0; 
    }
    
    for (int i = 1; i <= n; i++) potential[i] = dist1[i];
    
    for (int vtx = n; vtx != 1; vtx = par1[vtx][0]) used[par1[vtx][1]] = 1;

    for (int i = 0; i < m; i++) {
        long long rc = c[i] + potential[u[i]] - potential[v[i]];
        if (!used[i]) adj2[u[i]][deg2[u[i]]++] = (edge){v[i], rc, i, 0};
        else adj2[v[i]][deg2[v[i]]++] = (edge){u[i], -rc, i, 1};
    }

    for (int i = 1; i <= n; i++) dist2[i] = INT_MAX;

    dist2[1] = 0;
    for (int it = 0; it < n-1; it++) {
        for (int u0 = 1; u0 <= n; u0++) {
            if (dist2[u0] == INT_MAX) {
                continue;
            }
            for (int k = 0; k < deg2[u0]; k++) {
                edge e = adj2[u0][k];
                if (dist2[u0] + e.cost < dist2[e.to])
                    dist2[e.to] = dist2[u0] + e.cost;
            }
        }
    }
    if (dist2[n] >= INT_MAX) { 
        printf("NOT_FEASIBLE"); 
        return 0; 
    }

    long long ans = 2 * potential[n] + dist2[n];
    printf("%lld", ans);
    return 0;
}