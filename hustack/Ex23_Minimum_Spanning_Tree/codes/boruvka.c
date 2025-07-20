#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MAXM 100005
#define MAXN 100005

int parent[MAXN];
int rank[MAXN];
int n, m;

typedef struct {
    int w, u, v;
} Edge;

Edge edges[MAXM];

int find(int u) {
    while (u != parent[u]) {
        u = parent[u];
    }
    return u;
}

int union_set(int u, int v) {
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
    scanf("%d %d", &n, &m);
    for (int i = 0; i < m; ++i) {
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].w);
    }

    for (int i = 1; i <= n; ++i) {
        parent[i] = i;
        rank[i] = 0;
    }

    int total = 0;
    int num_comp = n;

    while (num_comp > 1) {
        Edge* cheapest[MAXN] = {NULL};

        for (int i = 0; i < m; ++i) {
            int u = edges[i].u;
            int v = edges[i].v;
            int w = edges[i].w;
            int ru = find(u);
            int rv = find(v);
            if (ru == rv) continue;

            if (cheapest[ru] == NULL || cheapest[ru] -> w > w)
                cheapest[ru] = &edges[i];
            if (cheapest[rv] == NULL || cheapest[rv] -> w > w)
                cheapest[rv] = &edges[i];
        }

        for (int i = 1; i <= n; ++i) {
            if (cheapest[i] != NULL) {
                int u = cheapest[i] -> u;
                int v = cheapest[i] -> v;
                int w = cheapest[i] -> w;
                if (union_set(u, v)) {
                    total += w;
                    num_comp--;
                }
            }
        }
    }

    printf("%d", total);
    return 0;
}
