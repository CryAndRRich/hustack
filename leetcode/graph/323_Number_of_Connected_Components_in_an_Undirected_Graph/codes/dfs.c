#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_N 2000

void dfs(int node, bool* visited, int** graph, int* graphSize) {
    visited[node] = true;
    for (int i = 0; i < graphSize[node]; i++) {
        int neighbor = graph[node][i];
        if (!visited[neighbor]) {
            dfs(neighbor, visited, graph, graphSize);
        }
    }
}

int countComponents(int n, int** edges, int edgesSize, int* edgesColSize) {
    int** graph = (int**)malloc(sizeof(int*) * n);
    int* graphSize = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        graph[i] = (int*)malloc(sizeof(int) * MAX_N);
    }

    for (int i = 0; i < edgesSize; i++) {
        int u = edges[i][0];
        int v = edges[i][1];
        graph[u][graphSize[u]++] = v;
        graph[v][graphSize[v]++] = u;
    }

    bool* visited = (bool*)calloc(n, sizeof(bool));
    int count = 0;

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            dfs(i, visited, graph, graphSize);
            count++;
        }
    }

    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
    free(graphSize);
    free(visited);

    return count;
}
