#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MAXV 35
#define MAXE 70  

typedef struct { 
    int to, rev; 
    int cap; 
    long long cost; 
} edge;

int n;
static edge g[MAXV][MAXE];
static int g_sz[MAXV];
long long dist[MAXV], potential[MAXV];
int pv_v[MAXV], pv_e[MAXV];

static inline void addEdge(int u, int v, int cap, long long cost) {
    g[u][g_sz[u]] = (edge){v, g_sz[v], cap, cost};
    g[v][g_sz[v]] = (edge){u, g_sz[u], 0, -cost};
    g_sz[u]++;
    g_sz[v]++;
}

static inline long long minCostMaxFlow(int s, int t, int maxf, int *out_flow) {
    int flow = 0;
    long long cost = 0;
    for (int i = 0; i <= n; i++) potential[i] = 0;
    while (flow < maxf) {
        for (int i = 0; i <= n; i++) dist[i] = INT_MAX;
        dist[s] = 0;
        int used[MAXV] = {0};
        while (1) {
            int u = -1;
            long long best = INT_MAX;
            for (int i = 0; i <= n; i++) if (!used[i] && dist[i] < best) {
                best = dist[i]; u = i;
            }
            if (u < 0) break;
            used[u] = 1;
            for (int ei = 0; ei < g_sz[u]; ei++) {
                edge *e = &g[u][ei];
                if (e -> cap <= 0) continue;
                long long nd = dist[u] + e -> cost + potential[u] - potential[e -> to];
                if (nd < dist[e -> to]) {
                    dist[e -> to] = nd;
                    pv_v[e -> to] = u;
                    pv_e[e -> to] = ei;
                }
            }
        }
        if (dist[t] == INT_MAX) break;
        for (int i = 0; i <= n; i++) if (dist[i] < INT_MAX) potential[i] += dist[i];
        int addf = maxf - flow;
        int v = t;
        while (v != s) {
            edge *e = &g[pv_v[v]][pv_e[v]];
            addf = addf < e -> cap ? addf : e -> cap;
            v = pv_v[v];
        }
        flow += addf;
        cost += (long long)addf * potential[t];
        v = t;
        while (v != s) {
            edge *e = &g[pv_v[v]][pv_e[v]];
            e -> cap -= addf;
            g[v][e -> rev].cap += addf;
            v = pv_v[v];
        }
    }
    *out_flow = flow;
    return cost;
}

int main(){
    int m;
    scanf("%d %d", &n, &m);
    int u, v;
    long long c;
    for (int i = 0; i <= n; i++) g_sz[i] = 0;
    for (int i = 0; i < m; i++){
        scanf("%d %d %lld", &u, &v, &c);
        addEdge(u, v, 1, c);
    }
    int flow;
    long long cost = minCostMaxFlow(1, n, 2, &flow);
    if (flow < 2) {
        printf("NOT_FEASIBLE");
    } else {
        printf("%lld", cost);
    }
    return 0;
}