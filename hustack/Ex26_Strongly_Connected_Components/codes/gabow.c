#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 300005
#define MAXM 1000005

int N, M;
int head[MAXN], to[MAXM], nxt[MAXM], edge_idx = 0;

void add_edge(int u, int v) {
    to[edge_idx] = v;
    nxt[edge_idx] = head[u];
    head[u] = edge_idx++;
}

int dfs_num[MAXN], dfs_clock = 0;
int S[MAXN], P[MAXN], top_s = 0, top_p = 0;
int in_stack[MAXN];  
int scc_count = 0;

void gabow(int u) {
    dfs_num[u] = ++dfs_clock;
    S[top_s++] = u;
    P[top_p++] = u;
    in_stack[u] = 1;

    for (int i = head[u]; i != -1; i = nxt[i]) {
        int v = to[i];
        if (dfs_num[v] == 0) {
            gabow(v);
        } else if (in_stack[v]) {
            while (dfs_num[P[top_p - 1]] > dfs_num[v]) {
                top_p--;
            }
        }
    }

    if (P[top_p - 1] == u) {
        top_p--;
        int v;
        do {
            v = S[--top_s];
            in_stack[v] = 0;
        } while (v != u);
        scc_count++;
    }
}

int main() {
    scanf("%d %d", &N, &M);
    memset(head, -1, sizeof(int) * N);

    for (int i = 0; i < M; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        add_edge(u - 1, v - 1);  
    }

    memset(dfs_num, 0, sizeof(int) * N);
    memset(in_stack, 0, sizeof(int) * N);

    for (int i = 0; i < N; i++) {
        if (dfs_num[i] == 0)
            gabow(i);
    }

    printf("%d", scc_count);
    return 0;
}
