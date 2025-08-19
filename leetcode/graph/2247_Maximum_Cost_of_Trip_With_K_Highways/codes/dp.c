#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int *to;
    int *cost;
    int size;
    int capacity;
} Graph;

void addEdge(Graph* g, int u, int v, int c) {
    if (g[u].size == g[u].capacity) {
        g[u].capacity = g[u].capacity ? g[u].capacity * 2 : 2;
        g[u].to = (int*)realloc(g[u].to, sizeof(int)*g[u].capacity);
        g[u].cost = (int*)realloc(g[u].cost, sizeof(int)*g[u].capacity);
    }
    g[u].to[g[u].size] = v;
    g[u].cost[g[u].size] = c;
    g[u].size++;
}

int maximumCost(int n, int** highways, int highwaysSize, int* highwaysColSize, int k) {
    if (k + 1 > n) return -1;

    Graph *graph = (Graph*)calloc(n, sizeof(Graph));
    for (int i = 0; i < highwaysSize; i++) {
        int u = highways[i][0], v = highways[i][1], c = highways[i][2];
        addEdge(graph, u, v, c);
        addEdge(graph, v, u, c);
    }

    int maxMask = 1 << n;
    int **dp = (int**)malloc(maxMask * sizeof(int*));
    for (int i = 0; i < maxMask; i++) {
        dp[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) dp[i][j] = -1;
    }

    for (int i = 0; i < n; i++) {
        dp[1 << i][i] = 0;
    }

    int ans = -1;
    for (int mask = 0; mask < maxMask; mask++) {
        // count bits in mask
        int count = 0;
        int x = mask;
        while (x) {
            count += x & 1;
            x >>= 1;
        }
        if (count > k + 1) continue;
        for (int u = 0; u < n; u++) {
            if (dp[mask][u] == -1) continue;
            if (count == k + 1) {
                if (dp[mask][u] > ans) ans = dp[mask][u];
                continue;
            }
            for (int i = 0; i < graph[u].size; i++) {
                int v = graph[u].to[i];
                int c = graph[u].cost[i];
                if ((mask & (1 << v)) == 0) {
                    int newMask = mask | (1 << v);
                    if (dp[newMask][v] < dp[mask][u] + c) {
                        dp[newMask][v] = dp[mask][u] + c;
                    }
                }
            }
        }
    }

    for (int i = 0; i < maxMask; i++) free(dp[i]);
    free(dp);

    for (int i = 0; i < n; i++) {
        free(graph[i].to);
        free(graph[i].cost);
    }
    free(graph);

    return ans;
}
