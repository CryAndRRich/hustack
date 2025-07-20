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
int head[MAXV], edge_cnt = 0;

int height[MAXV], excess[MAXV], seen[MAXV];
int n, m;

void addEdge(int from, int to, int cap) {
    edges[edge_cnt] = (Edge){to, edge_cnt + 1, cap, head[from]};
    head[from] = edge_cnt++;
    edges[edge_cnt] = (Edge){from, edge_cnt - 1, 0, head[to]};
    head[to] = edge_cnt++;
}

void push(int u, Edge* e) {
    int v = e -> to;
    int d = height[u] - height[v];
    if (d == 1 && e -> cap > 0) {
        int flow = excess[u] < e -> cap ? excess[u] : e -> cap;
        e -> cap -= flow;
        edges[e -> rev].cap += flow;
        excess[u] -= flow;
        excess[v] += flow;
    }
}

void relabel(int u) {
    int min_height = INT_MAX;
    for (int i = head[u]; i != -1; i = edges[i].next) {
        if (edges[i].cap > 0) {
            int v = edges[i].to;
            if (height[v] < min_height) {
                min_height = height[v];
            }
        }
    }
    if (min_height < INT_MAX)
        height[u] = min_height + 1;
}

void discharge(int u, int V) {
    while (excess[u] > 0) {
        if (seen[u] == -1) seen[u] = head[u];
        int done = 1;
        for (int i = seen[u]; i != -1; i = edges[i].next) {
            seen[u] = i;
            Edge* e = &edges[i];
            if (e -> cap > 0 && height[u] == height[e -> to] + 1) {
                push(u, e);
                done = 0;
                break;
            }
        }
        if (done) {
            relabel(u);
            seen[u] = -1;
        }
    }
}

int maxFlow(int s, int t, int V) {
    memset(height, 0, sizeof(height));
    memset(excess, 0, sizeof(excess));
    memset(seen, -1, sizeof(seen));

    height[s] = V;
    for (int i = head[s]; i != -1; i = edges[i].next) {
        Edge* e = &edges[i];
        int flow = e -> cap;
        e -> cap -= flow;
        edges[e -> rev].cap += flow;
        excess[e -> to] += flow;
        excess[s] -= flow;
    }

    int list[MAXV], nlist = 0;
    for (int i = 0; i < V; ++i)
        if (i != s && i != t) list[nlist++] = i;

    int p = 0;
    while (p < nlist) {
        int u = list[p];
        int old_height = height[u];
        discharge(u, V);
        if (height[u] > old_height) {
            int temp = list[p];
            for (int j = p; j > 0; --j) list[j] = list[j - 1];
            list[0] = temp;
            p = 0;
        } else {
            ++p;
        }
    }

    return excess[t];
}

int main() {
    memset(head, -1, sizeof(head));
    scanf("%d %d", &n, &m);
    int S = 0, T = n + m + 1;
    int V = T + 1;

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

    printf("%d", maxFlow(S, T, V));
    return 0;
}
