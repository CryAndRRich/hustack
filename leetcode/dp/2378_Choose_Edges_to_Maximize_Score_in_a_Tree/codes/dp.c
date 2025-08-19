#include <stdio.h>
#include <stdlib.h>

typedef struct { int v, w; } Edge;

typedef struct {
    Edge* arr;
    int size;
} Vec;

Vec* g;
long long** memo;
char** vis;

long long dp(int u, int used) {
    if (vis[u][used]) return memo[u][used];
    vis[u][used] = 1;
    long long res = 0;
    for (int i = 0; i < g[u].size; i++) {
        res += dp(g[u].arr[i].v, 0);
    }
    if (!used) {
        for (int i = 0; i < g[u].size; i++) {
            if (g[u].arr[i].w <= 0) continue;
            long long cur = g[u].arr[i].w + dp(g[u].arr[i].v, 1);
            for (int j = 0; j < g[u].size; j++) {
                if (j != i) cur += dp(g[u].arr[j].v, 0);
            }
            if (cur > res) res = cur;
        }
    }
    memo[u][used] = res;
    return res;
}

long long maxScore(int** edges, int edgesSize, int* edgesColSize) {
    g = calloc(edgesSize, sizeof(Vec));
    for (int i = 1; i < edgesSize; i++) {
        int p = edges[i][0], w = edges[i][1];
        g[p].arr = realloc(g[p].arr, sizeof(Edge) * (g[p].size + 1));
        g[p].arr[g[p].size++] = (Edge){i, w};
    }
    memo = malloc(edgesSize * sizeof(long long*));
    vis = malloc(edgesSize * sizeof(char*));
    for (int i = 0; i < edgesSize; i++) {
        memo[i] = calloc(2, sizeof(long long));
        vis[i] = calloc(2, sizeof(char));
    }
    long long ans = dp(0, 0);
    for (int i = 0; i < edgesSize; i++) {
        free(g[i].arr);
        free(memo[i]);
        free(vis[i]);
    }
    free(g);
    free(memo);
    free(vis);
    return ans;
}
