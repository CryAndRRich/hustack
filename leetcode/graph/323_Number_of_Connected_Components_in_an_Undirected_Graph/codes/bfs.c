#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_N 2000

int countComponents(int n, int** edges, int edgesSize, int* edgesColSize) {
    int** graph = (int**)malloc(n * sizeof(int*));
    int* graphSize = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        graph[i] = (int*)malloc(MAX_N * sizeof(int));
    }

    // Build graph
    for (int i = 0; i < edgesSize; i++) {
        int u = edges[i][0];
        int v = edges[i][1];
        graph[u][graphSize[u]++] = v;
        graph[v][graphSize[v]++] = u;
    }

    bool* visited = (bool*)calloc(n, sizeof(bool));
    int* queue = (int*)malloc(n * sizeof(int));
    int count = 0;

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            int front = 0, back = 0;
            queue[back++] = i;
            visited[i] = true;

            while (front < back) {
                int node = queue[front++];
                for (int j = 0; j < graphSize[node]; j++) {
                    int neighbor = graph[node][j];
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue[back++] = neighbor;
                    }
                }
            }

            count++;
        }
    }

    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
    free(graphSize);
    free(visited);
    free(queue);

    return count;
}
