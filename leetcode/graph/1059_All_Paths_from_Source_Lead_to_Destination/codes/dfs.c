#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX 10001

int graph[MAX][MAX];
int graphSize[MAX];
int visited[MAX]; 

bool dfs(int node, int destination) {
    if (visited[node] == 1) return false;
    if (visited[node] == 2) return true;
    if (graphSize[node] == 0) return node == destination;

    visited[node] = 1;
    for (int i = 0; i < graphSize[node]; ++i) {
        if (!dfs(graph[node][i], destination)) return false;
    }
    visited[node] = 2;
    return true;
}

bool leadsToDestination(int n, int** edges, int edgesSize, int* edgesColSize, int source, int destination) {
    for (int i = 0; i < n; i++) {
        graphSize[i] = 0;
        visited[i] = 0;
    }

    for (int i = 0; i < edgesSize; i++) {
        int u = edges[i][0], v = edges[i][1];
        graph[u][graphSize[u]++] = v;
    }

    return dfs(source, destination);
}
