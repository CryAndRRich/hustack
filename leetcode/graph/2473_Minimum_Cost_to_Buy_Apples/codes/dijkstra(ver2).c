#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct {
    int v, w;
} Edge;

typedef struct {
    Edge* edges;
    int size, cap;
} AdjList;

typedef struct {
    long long cost;
    int city;
} HeapNode;

typedef struct {
    HeapNode* arr;
    int size, cap;
} MinHeap;

MinHeap* createHeap(int cap) {
    MinHeap* h = (MinHeap*)malloc(sizeof(MinHeap));
    h->arr = (HeapNode*)malloc(sizeof(HeapNode) * cap);
    h->size = 0; h->cap = cap;
    return h;
}

void swapNode(HeapNode* a, HeapNode* b) {
    HeapNode t = *a; *a = *b; *b = t;
}

void pushHeap(MinHeap* h, long long cost, int city) {
    int i = h->size++;
    h->arr[i].cost = cost;
    h->arr[i].city = city;
    while (i && h->arr[i].cost < h->arr[(i - 1) / 2].cost) {
        swapNode(&h->arr[i], &h->arr[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

HeapNode popHeap(MinHeap* h) {
    HeapNode root = h->arr[0];
    h->arr[0] = h->arr[--h->size];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, smallest = i;
        if (l < h->size && h->arr[l].cost < h->arr[smallest].cost) smallest = l;
        if (r < h->size && h->arr[r].cost < h->arr[smallest].cost) smallest = r;
        if (smallest == i) break;
        swapNode(&h->arr[i], &h->arr[smallest]);
        i = smallest;
    }
    return root;
}

void addEdge(AdjList* g, int u, int v, int w) {
    if (g[u].size == g[u].cap) {
        g[u].cap *= 2;
        g[u].edges = realloc(g[u].edges, g[u].cap * sizeof(Edge));
    }
    g[u].edges[g[u].size].v = v;
    g[u].edges[g[u].size].w = w;
    g[u].size++;
}

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
long long* minCost(int n, int** roads, int roadsSize, int* roadsColSize,
                   int* appleCost, int appleCostSize, int k, int* returnSize) {
    *returnSize = n;
    AdjList* graph = malloc(n * sizeof(AdjList));
    for (int i = 0; i < n; i++) {
        graph[i].size = 0;
        graph[i].cap = 2;
        graph[i].edges = malloc(2 * sizeof(Edge));
    }
    for (int i = 0; i < roadsSize; i++) {
        int a = roads[i][0] - 1, b = roads[i][1] - 1, c = roads[i][2];
        addEdge(graph, a, b, c);
        addEdge(graph, b, a, c);
    }

    long long* result = malloc(n * sizeof(long long));
    for (int i = 0; i < n; i++) result[i] = appleCost[i];

    MinHeap* heap = createHeap(n * n + 5);
    for (int i = 0; i < n; i++) pushHeap(heap, result[i], i);

    while (heap->size) {
        HeapNode node = popHeap(heap);
        long long cost = node.cost;
        int city = node.city;
        if (cost > result[city]) continue;
        for (int i = 0; i < graph[city].size; i++) {
            int nb = graph[city].edges[i].v;
            int w = graph[city].edges[i].w;
            long long newCost = result[city] + (long long)(k + 1) * w;
            if (newCost < result[nb]) {
                result[nb] = newCost;
                pushHeap(heap, newCost, nb);
            }
        }
    }

    for (int i = 0; i < n; i++) free(graph[i].edges);
    free(graph);
    free(heap->arr);
    free(heap);
    return result;
}
