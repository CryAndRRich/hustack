#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct Edge {
    int to, weight;
    struct Edge* next;
} Edge;

typedef struct {
    int node, dist;
} HeapNode;

typedef struct {
    HeapNode* data;
    int size;
    int capacity;
} MinHeap;

MinHeap* createHeap(int cap) {
    MinHeap* h = (MinHeap*)malloc(sizeof(MinHeap));
    h -> data = (HeapNode*)malloc(sizeof(HeapNode) * cap);
    h -> size = 0;
    h -> capacity = cap;
    return h;
}

void swap(HeapNode* a, HeapNode* b) {
    HeapNode tmp = *a; *a = *b; *b = tmp;
}

void heapPush(MinHeap* h, int node, int dist) {
    int i = h -> size++;
    h -> data[i].node = node;
    h -> data[i].dist = dist;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h -> data[p].dist <= h -> data[i].dist) break;
        swap(&h -> data[p], &h -> data[i]);
        i = p;
    }
}

int heapPop(MinHeap* h, HeapNode* out) {
    if (h -> size == 0) return 0;
    *out = h -> data[0];
    h -> data[0] = h -> data[--h -> size];
    int i = 0;
    while (1) {
        int l = 2*i+1, r = 2*i+2, smallest = i;
        if (l < h -> size && h -> data[l].dist < h -> data[smallest].dist) smallest = l;
        if (r < h -> size && h -> data[r].dist < h -> data[smallest].dist) smallest = r;
        if (smallest == i) break;
        swap(&h -> data[i], &h -> data[smallest]);
        i = smallest;
    }
    return 1;
}

int minimumDistance(int n, int** edges, int edgesSize, int* edgesColSize, int s, int* marked, int markedSize) {
    Edge** adj = (Edge**)calloc(n, sizeof(Edge*));
    for (int i = 0; i < edgesSize; i++) {
        Edge* e = (Edge*)malloc(sizeof(Edge));
        e -> to = edges[i][1];
        e -> weight = edges[i][2];
        e -> next = adj[edges[i][0]];
        adj[edges[i][0]] = e;
    }

    char* isMarked = (char*)calloc(n, sizeof(char));
    for (int i = 0; i < markedSize; i++) {
        isMarked[marked[i]] = 1;
    }

    int* dist = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) dist[i] = INT_MAX;
    dist[s] = 0;

    MinHeap* heap = createHeap(edgesSize + 5);
    heapPush(heap, s, 0);

    HeapNode cur;
    while (heapPop(heap, &cur)) {
        int node = cur.node;
        int d = cur.dist;
        if (d > dist[node]) continue;
        if (isMarked[node]) {
            return d;
        }
        for (Edge* e = adj[node]; e != NULL; e = e -> next) {
            int nd = d + e -> weight;
            if (nd < dist[e -> to]) {
                dist[e -> to] = nd;
                heapPush(heap, e -> to, nd);
            }
        }
    }
    return -1;
}
