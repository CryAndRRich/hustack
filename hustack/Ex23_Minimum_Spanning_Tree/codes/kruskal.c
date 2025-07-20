#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int u, v, w;
} Edge;

int cmp(const void *a, const void *b) {
    return ((Edge*)a)->w - ((Edge*)b)->w;
}

int parent[100005], rank[100005];

int find(int u) {
    while (u != parent[u]) {
        u = parent[u];
    }
    return u;
}

int union_sets(int u, int v) {
    int ru = find(u);
    int rv = find(v);
    if (ru == rv) return 0;
    if (rank[ru] < rank[rv]) {
        parent[ru] = rv;
    } else {
        parent[rv] = ru;
        if (rank[ru] == rank[rv]) {
            rank[ru]++;
        }
    }
    return 1;
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    Edge *edges = malloc(m * sizeof(Edge));
    for (int i = 0; i < m; ++i) {
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].w);
    }

    qsort(edges, m, sizeof(Edge), cmp);

    for (int i = 1; i <= n; ++i) parent[i] = i;

    long long total = 0;
    for (int i = 0; i < m; ++i) {
        if (union_sets(edges[i].u, edges[i].v)) {
            total += edges[i].w;
        }
    }

    printf("%lld", total);
    free(edges);
    return 0;
}
