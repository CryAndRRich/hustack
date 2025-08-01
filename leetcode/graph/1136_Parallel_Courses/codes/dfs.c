#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_N 5001

int graph[MAX_N][MAX_N];
int degree[MAX_N];
int visited[MAX_N];
int dp[MAX_N]; 

int dfs(int node, int n) {
    if (visited[node] == 1) return -1;
    if (visited[node] == 2) return dp[node];

    visited[node] = 1;
    int max_depth = 1;
    for (int i = 0; i < degree[node]; ++i) {
        int nei = graph[node][i];
        int res = dfs(nei, n);
        if (res == -1) return -1;
        if (res + 1 > max_depth)
            max_depth = res + 1;
    }
    visited[node] = 2;
    dp[node] = max_depth;
    return dp[node];
}

int minimumSemesters(int n, int** relations, int relationsSize, int* relationsColSize){
    memset(degree, 0, sizeof(degree));
    memset(visited, 0, sizeof(visited));
    memset(dp, 0, sizeof(dp));

    for (int i = 0; i < relationsSize; ++i) {
        int u = relations[i][0];
        int v = relations[i][1];
        graph[u][degree[u]++] = v;
    }

    int res = 0;
    for (int i = 1; i <= n; ++i) {
        int r = dfs(i, n);
        if (r == -1) return -1;
        if (r > res)
            res = r;
    }
    return res;
}
