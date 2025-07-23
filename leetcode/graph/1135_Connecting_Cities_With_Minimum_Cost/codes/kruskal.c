#include <stdio.h>
#include <stdlib.h>

#define MAXN 10005

int find(int* parent, int x) {
    if (parent[x] != x)
        parent[x] = find(parent, parent[x]);
    return parent[x];
}

int cmp(const void* a, const void* b) {
    const int* A = *(const int**)a;
    const int* B = *(const int**)b;
    return A[2] - B[2];
}

int minimumCost(int n, int** connections, int connectionsSize, int* connectionsColSize) {
    int parent[MAXN];
    for (int i = 1; i <= n; i++) parent[i] = i;

    qsort(connections, connectionsSize, sizeof(int*), cmp);

    int cost = 0, count = 0;
    for (int i = 0; i < connectionsSize; i++) {
        int u = connections[i][0];
        int v = connections[i][1];
        int w = connections[i][2];
        int pu = find(parent, u);
        int pv = find(parent, v);
        if (pu != pv) {
            parent[pu] = pv;
            cost += w;
            if (++count == n - 1)
                return cost;
        }
    }

    return -1;
}
