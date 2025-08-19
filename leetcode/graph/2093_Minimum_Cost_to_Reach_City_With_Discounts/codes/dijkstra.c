#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
    int to;
    int cost;
    int next;
} Edge;

typedef struct {
    Edge* edges;
    int* head;
    int edgeCount;
    int n;
    int capacity;
} Graph;

void initGraph(Graph* g, int n) {
    g -> n = n;
    g -> capacity = 1000;
    g -> edges = (Edge*)malloc(sizeof(Edge) * g -> capacity);
    g -> head = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) g -> head[i] = -1;
    g -> edgeCount = 0;
}

void addEdge(Graph* g, int u, int v, int cost) {
    if (g -> edgeCount == g -> capacity) {
        g -> capacity *= 2;
        g -> edges = (Edge*)realloc(g -> edges, sizeof(Edge) * g -> capacity);
    }
    g -> edges[g -> edgeCount] = (Edge){v, cost, g -> head[u]};
    g -> head[u] = g -> edgeCount++;
}

typedef struct {
    int cost;
    int node;
    int discounts_left;
} State;

typedef struct {
    State* data;
    int size;
    int capacity;
} MinHeap;

MinHeap* createHeap(int capacity) {
    MinHeap* h = (MinHeap*)malloc(sizeof(MinHeap));
    h -> data = (State*)malloc(sizeof(State)*capacity);
    h -> size = 0;
    h -> capacity = capacity;
    return h;
}

void swap(State* a, State* b) {
    State tmp = *a; *a = *b; *b = tmp;
}

void heapPush(MinHeap* h, State s) {
    if (h -> size == h -> capacity) {
        h -> capacity *= 2;
        h -> data = (State*)realloc(h -> data, sizeof(State)*h -> capacity);
    }
    int i = h -> size++;
    h -> data[i] = s;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h -> data[p].cost <= h -> data[i].cost) break;
        swap(&h -> data[p], &h -> data[i]);
        i = p;
    }
}

State heapPop(MinHeap* h) {
    State ret = h -> data[0];
    h -> data[0] = h -> data[--h -> size];
    int i = 0;
    while (1) {
        int left = i*2 + 1;
        int right = i*2 + 2;
        int smallest = i;
        if (left < h -> size && h -> data[left].cost < h -> data[smallest].cost) smallest = left;
        if (right < h -> size && h -> data[right].cost < h -> data[smallest].cost) smallest = right;
        if (smallest == i) break;
        swap(&h -> data[i], &h -> data[smallest]);
        i = smallest;
    }
    return ret;
}

int heapEmpty(MinHeap* h) {
    return h -> size == 0;
}

int minimumCost(int n, int** highways, int highwaysSize, int* highwaysColSize, int discounts) {
    Graph g;
    initGraph(&g, n);
    for (int i = 0; i < highwaysSize; i++) {
        int u = highways[i][0], v = highways[i][1], cost = highways[i][2];
        addEdge(&g, u, v, cost);
        addEdge(&g, v, u, cost);
    }
    
    int **dist = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++) {
        dist[i] = (int*)malloc(sizeof(int) * (discounts + 1));
        for (int j = 0; j <= discounts; j++) dist[i][j] = INT_MAX;
    }
    
    dist[0][discounts] = 0;
    MinHeap* heap = createHeap(1000);
    heapPush(heap, (State){0, 0, discounts});
    
    while (!heapEmpty(heap)) {
        State cur = heapPop(heap);
        int cost = cur.cost, node = cur.node, d_left = cur.discounts_left;
        if (node == n - 1) {
            // free memory
            for (int i = 0; i < n; i++) free(dist[i]);
            free(dist);
            free(g.edges);
            free(g.head);
            free(heap -> data);
            free(heap);
            return cost;
        }
        if (dist[node][d_left] < cost) continue;
        
        for (int i = g.head[node]; i != -1; i = g.edges[i].next) {
            int nxt = g.edges[i].to;
            int c = g.edges[i].cost;
            
            if (cost + c < dist[nxt][d_left]) {
                dist[nxt][d_left] = cost + c;
                heapPush(heap, (State){dist[nxt][d_left], nxt, d_left});
            }
            if (d_left > 0) {
                int cDiscount = cost + c / 2;
                if (cDiscount < dist[nxt][d_left - 1]) {
                    dist[nxt][d_left - 1] = cDiscount;
                    heapPush(heap, (State){cDiscount, nxt, d_left - 1});
                }
            }
        }
    }
    
    for (int i = 0; i < n; i++) free(dist[i]);
    free(dist);
    free(g.edges);
    free(g.head);
    free(heap -> data);
    free(heap);
    return -1;
}
