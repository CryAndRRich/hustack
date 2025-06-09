#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 300005
#define MAXM 1000005

int N, M;
int head[MAXN], to[MAXM], nxt[MAXM], idx;

int dfn[MAXN], low[MAXN], timer;
int stk[MAXN], top;
int in_stack[MAXN];
int scc_count;

void add_edge(int u, int v) {
    to[idx] = v;
    nxt[idx] = head[u];
    head[u] = idx++;
}

void tarjan(int u) {
    dfn[u] = low[u] = ++timer;
    stk[top++] = u;
    in_stack[u] = 1;

    for (int e = head[u]; e != -1; e = nxt[e]) {
        int v = to[e];
        if (!dfn[v]) {
            tarjan(v);
            if (low[v] < low[u]) low[u] = low[v];
        } else if (in_stack[v] && dfn[v] < low[u]) {
            low[u] = dfn[v];
        }
    }

    if (dfn[u] == low[u]) {
        while (1) {
            int v = stk[--top];
            in_stack[v] = 0;
            if (v == u) break;
        }
        scc_count++;
    }
}

int main() {
    scanf("%d %d", &N, &M);
    memset(head, -1, sizeof(int) * N);
    idx = 0;

    for (int i = 0; i < M; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        add_edge(u - 1, v - 1);
    }

    memset(dfn, 0, sizeof(int) * N);
    memset(low, 0, sizeof(int) * N);
    memset(in_stack, 0, sizeof(int) * N);
    timer = top = scc_count = 0;

    for (int i = 0; i < N; i++) {
        if (!dfn[i]) tarjan(i);
    }

    printf("%d", scc_count);
    return 0;
}
