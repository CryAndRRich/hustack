#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    int size;
    int cap;
} Vec;

void push(Vec *v, int val) {
    if (v -> size == v -> cap) {
        v -> cap = v -> cap ? v -> cap * 2 : 2;
        v -> data = realloc(v -> data, sizeof(int) * v -> cap);
    }
    v -> data[v -> size++] = val;
}

int* bfs(int start, Vec *g, int n, int *far) {
    int *dist = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) dist[i] = -1;
    int *queue = malloc(sizeof(int) * n);
    int front = 0, back = 0;
    dist[start] = 0;
    queue[back++] = start;
    *far = start;
    while (front < back) {
        int u = queue[front++];
        *far = u;
        for (int i = 0; i < g[u].size; i++) {
            int v = g[u].data[i];
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                queue[back++] = v;
            }
        }
    }
    free(queue);
    return dist;
}

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* lastMarkedNodes(int** edges, int edgesSize, int* edgesColSize, int* returnSize) {
    int n = edgesSize + 1;
    Vec *g = malloc(sizeof(Vec) * n);
    for (int i = 0; i < n; i++) {
        g[i].data = NULL;
        g[i].size = 0;
        g[i].cap = 0;
    }
    for (int i = 0; i < edgesSize; i++) {
        int u = edges[i][0], v = edges[i][1];
        push(&g[u], v);
        push(&g[v], u);
    }

    int f1, f2;
    int *tmp = bfs(0, g, n, &f1);
    free(tmp);
    int *dist1 = bfs(f1, g, n, &f2);
    int *dist2 = bfs(f2, g, n, &f1);

    int *res = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        res[i] = dist1[i] > dist2[i] ? f1 : f2;
    }

    for (int i = 0; i < n; i++) free(g[i].data);
    free(g);
    free(dist1);
    free(dist2);

    *returnSize = n;
    return res;
}
