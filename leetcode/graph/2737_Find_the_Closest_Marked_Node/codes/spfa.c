#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    int to, w;
} Edge;

typedef struct {
    Edge *data;
    int size, cap;
} Vec;

void push(Vec *v, int to, int w) {
    if (v -> size == v -> cap) {
        v -> cap = v -> cap ? v -> cap * 2 : 4;
        v -> data = realloc(v -> data, v -> cap * sizeof(Edge));
    }
    v -> data[v -> size++] = (Edge){to, w};
}

int minimumDistance(int n, int** edges, int edgesSize, int* edgesColSize, int s, int* marked, int markedSize) {
    Vec *graph = calloc(n, sizeof(Vec));
    for (int i = 0; i < edgesSize; i++) push(&graph[edges[i][0]], edges[i][1], edges[i][2]);
    int *dist = malloc(n * sizeof(int));
    bool *inQueue = calloc(n, sizeof(bool));
    for (int i = 0; i < n; i++) dist[i] = INT_MAX;
    dist[s] = 0;
    int *q = malloc(n * sizeof(int)), front = 0, back = 0;
    q[back++] = s;
    inQueue[s] = true;
    while (front != back) {
        int u = q[front++];
        if (front == n) front = 0;
        inQueue[u] = false;
        for (int i = 0; i < graph[u].size; i++) {
            int v = graph[u].data[i].to, w = graph[u].data[i].w;
            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                if (!inQueue[v]) {
                    q[back++] = v;
                    if (back == n) back = 0;
                    inQueue[v] = true;
                }
            }
        }
    }
    int ans = INT_MAX;
    for (int i = 0; i < markedSize; i++) if (dist[marked[i]] < ans) ans = dist[marked[i]];
    for (int i = 0; i < n; i++) free(graph[i].data);
    free(graph); free(dist); free(inQueue); free(q);
    return ans == INT_MAX ? -1 : ans;
}
