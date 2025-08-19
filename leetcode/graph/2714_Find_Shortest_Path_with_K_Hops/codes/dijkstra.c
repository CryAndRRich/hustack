#include <limits.h>
#include <stdlib.h>

typedef struct {
    int t, u, h;
} Node;

typedef struct {
    Node* data;
    int size, cap;
} MinHeap;

void push(MinHeap* h, Node val) {
    if (h -> size == h -> cap) {
        h -> cap *= 2;
        h -> data = realloc(h -> data, h -> cap * sizeof(Node));
    }
    int i = h -> size++;
    h -> data[i] = val;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h -> data[p].t <= h -> data[i].t) break;
        Node tmp = h -> data[p];
        h -> data[p] = h -> data[i];
        h -> data[i] = tmp;
        i = p;
    }
}

Node pop(MinHeap* h) {
    Node res = h -> data[0];
    h -> data[0] = h -> data[--h -> size];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < h -> size && h -> data[l].t < h -> data[s].t) s = l;
        if (r < h -> size && h -> data[r].t < h -> data[s].t) s = r;
        if (s == i) break;
        Node tmp = h -> data[s];
        h -> data[s] = h -> data[i];
        h -> data[i] = tmp;
        i = s;
    }
    return res;
}

int shortestPathWithHops(int n, int** edges, int edgesSize, int* edgesColSize, int s, int d, int k) {
    int** adj = malloc(n * sizeof(int*));
    int** w = malloc(n * sizeof(int*));
    int* deg = calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        adj[i] = malloc(edgesSize * sizeof(int));
        w[i] = malloc(edgesSize * sizeof(int));
    }
    for (int i = 0; i < edgesSize; i++) {
        int u = edges[i][0], v = edges[i][1], wt = edges[i][2];
        adj[u][deg[u]] = v; w[u][deg[u]++] = wt;
        adj[v][deg[v]] = u; w[v][deg[v]++] = wt;
    }
    int** dist = malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        dist[i] = malloc((k + 1) * sizeof(int));
        for (int j = 0; j <= k; j++) dist[i][j] = INT_MAX;
    }
    dist[s][0] = 0;
    MinHeap heap = {malloc(1024 * sizeof(Node)), 0, 1024};
    push(&heap, (Node){0, s, 0});
    while (heap.size) {
        Node cur = pop(&heap);
        if (cur.u == d) return cur.t;
        if (cur.t > dist[cur.u][cur.h]) continue;
        for (int idx = 0; idx < deg[cur.u]; idx++) {
            int v = adj[cur.u][idx], wt = w[cur.u][idx];
            if (cur.t + wt < dist[v][cur.h]) {
                dist[v][cur.h] = cur.t + wt;
                push(&heap, (Node){dist[v][cur.h], v, cur.h});
            }
            if (cur.h < k && cur.t < dist[v][cur.h + 1]) {
                dist[v][cur.h + 1] = cur.t;
                push(&heap, (Node){cur.t, v, cur.h + 1});
            }
        }
    }
    return -1;
}
