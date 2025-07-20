#include <stdbool.h>

int find(int* parent, int x) {
    if (parent[x] != x)
        parent[x] = find(parent, parent[x]);
    return parent[x];
}

bool unionSet(int* parent, int x, int y) {
    int px = find(parent, x), py = find(parent, y);
    if (px == py) return false;
    parent[px] = py;
    return true;
}

bool validTree(int n, int** edges, int edgesSize, int* edgesColSize) {
    if (edgesSize != n - 1) return false;

    int* parent = malloc(sizeof(int) * n);
    for (int i = 0; i < n; ++i) parent[i] = i;

    for (int i = 0; i < edgesSize; ++i) {
        int u = edges[i][0], v = edges[i][1];
        if (!unionSet(parent, u, v)) {
            free(parent);
            return false;
        }
    }

    free(parent);
    return true;
}
