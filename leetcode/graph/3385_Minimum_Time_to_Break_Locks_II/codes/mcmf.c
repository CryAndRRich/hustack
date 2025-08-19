#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define INF 1000000000
#define MAXN 250   
#define MAXE 50000 

typedef struct {
    int to, cap, cost, rev;
} Edge;

Edge adj[MAXN][MAXN * 2]; 
int edge_count[MAXN];

void add_edge(int u, int v, int cap, int cost) {
    adj[u][edge_count[u]].to = v;
    adj[u][edge_count[u]].cap = cap;
    adj[u][edge_count[u]].cost = cost;
    adj[u][edge_count[u]].rev = edge_count[v];
    edge_count[u]++;

    adj[v][edge_count[v]].to = u;
    adj[v][edge_count[v]].cap = 0;
    adj[v][edge_count[v]].cost = -cost;
    adj[v][edge_count[v]].rev = edge_count[u] - 1;
    edge_count[v]++;
}

int min_cost_flow(int N, int s, int t, int maxf) {
    int prevv[MAXN], preve[MAXN];
    int h[MAXN], dist[MAXN];
    int inqueue[MAXN];
    int flow = 0, res = 0;

    memset(h, 0, sizeof(h));

    while (flow < maxf) {
        for (int i = 0; i < N; i++) dist[i] = INF;
        dist[s] = 0;
        memset(inqueue, 0, sizeof(inqueue));

        int queue[MAXN * 10];
        int qh = 0, qt = 0;
        queue[qt++] = s;
        inqueue[s] = 1;

        while (qh != qt) {
            int u = queue[qh++];
            if (qh == MAXN * 10) qh = 0;
            inqueue[u] = 0;

            for (int i = 0; i < edge_count[u]; i++) {
                Edge *e = &adj[u][i];
                if (e->cap > 0 && dist[e->to] > dist[u] + e->cost + h[u] - h[e->to]) {
                    dist[e->to] = dist[u] + e->cost + h[u] - h[e->to];
                    prevv[e->to] = u;
                    preve[e->to] = i;
                    if (!inqueue[e->to]) {
                        queue[qt++] = e->to;
                        if (qt == MAXN * 10) qt = 0;
                        inqueue[e->to] = 1;
                    }
                }
            }
        }

        if (dist[t] == INF) return res;

        for (int i = 0; i < N; i++) h[i] += dist[i];

        int d = maxf - flow;
        for (int v = t; v != s; v = prevv[v]) {
            if (d > adj[prevv[v]][preve[v]].cap)
                d = adj[prevv[v]][preve[v]].cap;
        }

        flow += d;
        res += d * h[t];
        for (int v = t; v != s; v = prevv[v]) {
            Edge *e = &adj[prevv[v]][preve[v]];
            e->cap -= d;
            adj[v][e->rev].cap += d;
        }
    }
    return res;
}

int findMinimumTime(int* strength, int strengthSize) {
    int n = strengthSize;
    int source = 2 * n;
    int sink = 2 * n + 1;
    int N = 2 * n + 2;

    memset(edge_count, 0, sizeof(edge_count));

    for (int i = 0; i < n; i++) {
        add_edge(source, i, 1, 0);
        add_edge(n + i, sink, 1, 0);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int cost = (strength[j] + i) / (i + 1);
            add_edge(i, n + j, 1, cost);
        }
    }

    return min_cost_flow(N, source, sink, n);
}
