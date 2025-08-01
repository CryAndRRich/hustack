#include <stdio.h>
#include <stdlib.h>

int find(int* parent, int x) {
    if (parent[x] != x)
        parent[x] = find(parent, parent[x]);
    return parent[x];
}

void unionSet(int* parent, int x, int y) {
    int px = find(parent, x);
    int py = find(parent, y);
    if (px != py)
        parent[px] = py;
}

int countComponents(int n, int** edges, int edgesSize, int* edgesColSize) {
    int* parent = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) parent[i] = i;

    for (int i = 0; i < edgesSize; i++) {
        int u = edges[i][0];
        int v = edges[i][1];
        unionSet(parent, u, v);
    }

    int* seen = (int*)calloc(n, sizeof(int));
    int count = 0;
    for (int i = 0; i < n; i++) {
        int root = find(parent, i);
        if (!seen[root]) {
            seen[root] = 1;
            count++;
        }
    }

    free(parent);
    free(seen);
    return count;
}
