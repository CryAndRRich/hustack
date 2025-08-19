#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

typedef struct {
    int node;
    int mask;
    int cost;
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
    State t = *a;
    *a = *b;
    *b = t;
}

void heapPush(MinHeap* h, State s) {
    if (h -> size == h -> capacity) {
        h -> capacity *= 2;
        h -> data = (State*)realloc(h -> data, sizeof(State)*h -> capacity);
    }
    int i = h -> size++;
    h -> data[i] = s;
    while (i > 0) {
        int p = (i-1)/2;
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
        int left = i*2 + 1, right = i*2 + 2, smallest = i;
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

typedef struct {
    int *to;
    int *cost;
    int size;
    int capacity;
} Graph;

void addEdge(Graph* g, int u, int v, int c) {
    if (g[u].size == g[u].capacity) {
        g[u].capacity = g[u].capacity ? g[u].capacity * 2 : 2;
        g[u].to = (int*)realloc(g[u].to, sizeof(int)*g[u].capacity);
        g[u].cost = (int*)realloc(g[u].cost, sizeof(int)*g[u].capacity);
    }
    g[u].to[g[u].size] = v;
    g[u].cost[g[u].size] = c;
    g[u].size++;
}

int bitCount(int x) {
    int c = 0;
    while (x) {
        c += x & 1;
        x >>= 1;
    }
    return c;
}

int maximumCost(int n, int** highways, int highwaysSize, int* highwaysColSize, int k) {
    if (k + 1 > n) return -1;

    Graph* graph = (Graph*)calloc(n, sizeof(Graph));
    for (int i = 0; i < highwaysSize; i++) {
        int u = highways[i][0], v = highways[i][1], c = highways[i][2];
        addEdge(graph, u, v, -c);
        addEdge(graph, v, u, -c);
    }

    int maxMask = 1 << n;
    int** dist = (int**)malloc(maxMask * sizeof(int*));
    for (int i = 0; i < maxMask; i++) {
        dist[i] = (int*)malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) dist[i][j] = INT_MAX;
    }

    int ans = INT_MAX;
    for (int start = 0; start < n; start++) {
        MinHeap* heap = createHeap(1000);
        int startMask = 1 << start;
        dist[startMask][start] = 0;
        heapPush(heap, (State){start, startMask, 0});

        while (!heapEmpty(heap)) {
            State cur = heapPop(heap);
            if (dist[cur.mask][cur.node] < cur.cost) continue;

            if (bitCount(cur.mask) == k + 1) {
                if (cur.cost < ans) ans = cur.cost;
                break;
            }

            Graph g = graph[cur.node];
            for (int i = 0; i < g.size; i++) {
                int nxt = g.to[i];
                int c = g.cost[i];
                if ((cur.mask & (1 << nxt)) == 0) {
                    int nmask = cur.mask | (1 << nxt);
                    int ncost = cur.cost + c;
                    if (ncost < dist[nmask][nxt]) {
                        dist[nmask][nxt] = ncost;
                        heapPush(heap, (State){nxt, nmask, ncost});
                    }
                }
            }
        }
        for (int i = 0; i < maxMask; i++) {
            for (int j = 0; j < n; j++) {
                dist[i][j] = INT_MAX;
            }
        }
        free(heap -> data);
        free(heap);
    }

    for (int i = 0; i < maxMask; i++) free(dist[i]);
    free(dist);

    for (int i = 0; i < n; i++) {
        free(graph[i].to);
        free(graph[i].cost);
    }
    free(graph);

    return ans == INT_MAX ? -1 : -ans;
}
