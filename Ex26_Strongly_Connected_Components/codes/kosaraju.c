#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 300005
#define MAXM 1000005

int N, M;
int head[MAXN], head_rev[MAXN];
int to[MAXM], nxt[MAXM], to_rev[MAXM], nxt_rev[MAXM];
int idx = 0, idx_rev = 0;

int visited[MAXN], stack[MAXN], top = 0;

void add_edge(int u, int v) {
    to[idx] = v;
    nxt[idx] = head[u];
    head[u] = idx++;
}

void add_edge_rev(int u, int v) {
    to_rev[idx_rev] = v;
    nxt_rev[idx_rev] = head_rev[u];
    head_rev[u] = idx_rev++;
}

void dfs(int u) {
    visited[u] = 1;
    for (int i = head[u]; i != -1; i = nxt[i]) {
        int v = to[i];
        if (!visited[v]) dfs(v);
    }
    stack[top++] = u;
}

void rev_dfs(int u) {
    visited[u] = 1;
    for (int i = head_rev[u]; i != -1; i = nxt_rev[i]) {
        int v = to_rev[i];
        if (!visited[v]) rev_dfs(v);
    }
}

int count_scc() {
    memset(visited, 0, sizeof(int) * N);
    top = 0;

    for (int i = 0; i < N; i++)
        if (!visited[i]) dfs(i);

    memset(visited, 0, sizeof(int) * N);
    int scc = 0;

    for (int i = top - 1; i >= 0; i--) {
        int u = stack[i];
        if (!visited[u]) {
            rev_dfs(u);
            scc++;
        }
    }

    return scc;
}

int main() {
    scanf("%d %d", &N, &M);

    memset(head, -1, sizeof(int) * N);
    memset(head_rev, -1, sizeof(int) * N);

    for (int i = 0; i < M; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        add_edge(u - 1, v - 1);
        add_edge_rev(v - 1, u - 1);
    }

    printf("%d", count_scc());

    return 0;
}
