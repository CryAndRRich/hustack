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
int head[MAXV], level[MAXV], iter[MAXV];
int n, m, edge_cnt = 0;

void addEdge(int from, int to, int cap) {
    edges[edge_cnt] = (Edge){to, edge_cnt + 1, cap, head[from]};
    head[from] = edge_cnt++;
    edges[edge_cnt] = (Edge){from, edge_cnt - 1, 0, head[to]};
    head[to] = edge_cnt++;
}

int queue[MAXV];
int bfs(int s, int t) {
    memset(level, -1, sizeof(level));
    int front = 0, back = 0;
    queue[back++] = s;
    level[s] = 0;
    while (front < back) {
        int v = queue[front++];
        for (int i = head[v]; i != -1; i = edges[i].next) {
            Edge* e = &edges[i];
            if (e -> cap > 0 && level[e -> to] == -1) {
                level[e -> to] = level[v] + 1;
                queue[back++] = e -> to;
            }
        }
    }
    return level[t] != -1;
}

int dfs(int v, int t, int f) {
    if (v == t) return f;
    for (int* i = &iter[v]; *i != -1; *i = edges[*i].next) {
        Edge* e = &edges[*i];
        if (e -> cap > 0 && level[v] < level[e -> to]) {
            int d = dfs(e -> to, t, f < e -> cap ? f : e -> cap);
            if (d > 0) {
                e -> cap -= d;
                edges[e -> rev].cap += d;
                return d;
            }
        }
    }
    return 0;
}

int maxFlow(int s, int t) {
    int flow = 0;
    while (bfs(s, t)) {
        memcpy(iter, head, sizeof(head));
        int f;
        while ((f = dfs(s, t, INT_MAX)) > 0)
            flow += f;
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
