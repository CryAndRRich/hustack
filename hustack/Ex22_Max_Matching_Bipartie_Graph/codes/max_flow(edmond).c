#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#define MAXV 20001     
#define MAXE 1000000   

typedef struct {
    int to, rev, cap, next;
} Edge;

Edge edges[MAXE];
int head[MAXV];
int n, m, edge_cnt = 0;

void addEdge(int from, int to, int cap) {
    edges[edge_cnt] = (Edge){to, edge_cnt + 1, cap, head[from]};
    head[from] = edge_cnt++;
    edges[edge_cnt] = (Edge){from, edge_cnt - 1, 0, head[to]};
    head[to] = edge_cnt++;
}

int queue[MAXV];
int parent[MAXV];     
int parentEdge[MAXV];  

int bfs(int s, int t) {
    memset(parent, -1, sizeof(parent));
    int front = 0, back = 0;
    queue[back++] = s;
    parent[s] = -2;

    while (front < back) {
        int u = queue[front++];
        for (int i = head[u]; i != -1; i = edges[i].next) {
            Edge* e = &edges[i];
            int v = e->to;
            if (e->cap > 0 && parent[v] == -1) {
                parent[v] = u;
                parentEdge[v] = i;
                if (v == t) return 1;
                queue[back++] = v;
            }
        }
    }
    return 0;
}

int maxFlow(int s, int t) {
    int flow = 0;
    while (bfs(s, t)) {
        int path_flow = INT_MAX;
        for (int v = t; v != s; v = parent[v]) {
            int i = parentEdge[v];
            if (edges[i].cap < path_flow)
                path_flow = edges[i].cap;
        }

        for (int v = t; v != s; v = parent[v]) {
            int i = parentEdge[v];
            edges[i].cap -= path_flow;
            edges[edges[i].rev].cap += path_flow;
        }

        flow += path_flow;
    }
    return flow;
}

int main() {
    memset(head, -1, sizeof(head));
    scanf("%d %d", &n, &m);
    int S = 0, T = n + m + 1;

    for (int i = 1; i <= n; ++i) {
        addEdge(S, i, 1);
        int k;
        scanf("%d", &k);
        for (int j = 0; j < k; ++j) {
            int v;
            scanf("%d", &v);
            addEdge(i, n + v, 1);
        }
    }

    for (int i = 1; i <= m; ++i)
        addEdge(n + i, T, 1);

    printf("%d", maxFlow(S, T));
    return 0;
}
