#include <stdio.h>
#include <stdlib.h>

#define MAXN 10010

int parent[MAXN];

int find(int x) {
    if (parent[x] != x)
        parent[x] = find(parent[x]);
    return parent[x];
}

int unionSet(int x, int y) {
    int px = find(x), py = find(y);
    if (px == py) return 0;
    parent[px] = py;
    return 1;
}

typedef struct {
    int u, v, cost;
} Edge;

int cmp(const void* a, const void* b) {
    return ((Edge*)a)->cost - ((Edge*)b)->cost;
}

int minCostToSupplyWater(int n, int* wells, int wellsSize, int** pipes, int pipesSize, int* pipesColSize) {
    Edge* edges = malloc(sizeof(Edge) * (pipesSize + wellsSize));
    int edgeCount = 0;

    for (int i = 0; i < n; i++) {
        edges[edgeCount++] = (Edge){0, i + 1, wells[i]};
    }

    for (int i = 0; i < pipesSize; i++) {
        edges[edgeCount++] = (Edge){pipes[i][0], pipes[i][1], pipes[i][2]};
    }

    for (int i = 0; i <= n; i++) parent[i] = i;

    qsort(edges, edgeCount, sizeof(Edge), cmp);

    int cost = 0;
    for (int i = 0; i < edgeCount; i++) {
        if (unionSet(edges[i].u, edges[i].v)) {
            cost += edges[i].cost;
        }
    }

    free(edges);
    return cost;
}
