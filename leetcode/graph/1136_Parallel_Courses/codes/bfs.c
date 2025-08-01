#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int minimumSemesters(int n, int** relations, int relationsSize, int* relationsColSize) {
    int* in_degree = calloc(n + 1, sizeof(int));
    int** graph = malloc((n + 1) * sizeof(int*));
    int* graphSize = calloc(n + 1, sizeof(int));
    int* queue = malloc((n + 1) * sizeof(int));
    int front = 0, back = 0;

    for (int i = 0; i <= n; i++) {
        graph[i] = malloc((relationsSize + 1) * sizeof(int));
    }

    for (int i = 0; i < relationsSize; ++i) {
        int u = relations[i][0], v = relations[i][1];
        graph[u][graphSize[u]++] = v;
        in_degree[v]++;
    }

    for (int i = 1; i <= n; ++i)
        if (in_degree[i] == 0)
            queue[back++] = i;

    int semesters = 0, taken = 0;

    while (front < back) {
        int size = back - front;
        for (int i = 0; i < size; ++i) {
            int node = queue[front++];
            taken++;
            for (int j = 0; j < graphSize[node]; ++j) {
                int nei = graph[node][j];
                if (--in_degree[nei] == 0)
                    queue[back++] = nei;
            }
        }
        semesters++;
    }

    free(in_degree);
    free(graphSize);
    for (int i = 0; i <= n; i++) free(graph[i]);
    free(graph);
    free(queue);

    return taken == n ? semesters : -1;
}
