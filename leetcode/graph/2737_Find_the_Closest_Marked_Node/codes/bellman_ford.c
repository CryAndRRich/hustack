#include <limits.h>

int minimumDistance(int n, int** edges, int edgesSize, int* edgesColSize, int s, int* marked, int markedSize) {
    int dist[n];
    for (int i = 0; i < n; i++) dist[i] = INT_MAX;
    dist[s] = 0;

    for (int iter = 0; iter < n - 1; iter++) {
        for (int j = 0; j < edgesSize; j++) {
            int u = edges[j][0];
            int v = edges[j][1];
            int w = edges[j][2];
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }

    int minDist = INT_MAX;
    for (int i = 0; i < markedSize; i++) {
        if (dist[marked[i]] < minDist) {
            minDist = dist[marked[i]];
        }
    }

    return (minDist == INT_MAX) ? -1 : minDist;
}
