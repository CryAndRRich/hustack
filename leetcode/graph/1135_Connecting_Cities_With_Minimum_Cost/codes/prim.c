#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

#define MAXN 10005

typedef struct Edge {
    int to;
    int cost;
    struct Edge* next;
} Edge;

Edge* graph[MAXN];

void addEdge(int u, int v, int cost) {
    Edge* e1 = (Edge*)malloc(sizeof(Edge));
    e1 -> to = v;
    e1 -> cost = cost;
    e1 -> next = graph[u];
    graph[u] = e1;

    Edge* e2 = (Edge*)malloc(sizeof(Edge));
    e2 -> to = u;
    e2 -> cost = cost;
    e2 -> next = graph[v];
    graph[v] = e2;
}

typedef struct {
    int node;
    int cost;
} PQNode;

typedef struct {
    PQNode* data;
    int size;
} MinHeap;

void swap(PQNode* a, PQNode* b) {
    PQNode temp = *a;
    *a = *b;
    *b = temp;
}

MinHeap* createHeap(int capacity) {
    MinHeap* heap = (MinHeap*)malloc(sizeof(MinHeap));
    heap -> data = (PQNode*)malloc(sizeof(PQNode) * (capacity + 1));
    heap -> size = 0;
    return heap;
}

void heapPush(MinHeap* heap, int node, int cost) {
    int i = ++heap -> size;
    heap -> data[i].node = node;
    heap -> data[i].cost = cost;
    while (i > 1 && heap -> data[i].cost < heap -> data[i / 2].cost) {
        swap(&heap -> data[i], &heap -> data[i / 2]);
        i /= 2;
    }
}

PQNode heapPop(MinHeap* heap) {
    PQNode top = heap -> data[1];
    heap -> data[1] = heap -> data[heap -> size--];
    int i = 1;
    while (1) {
        int smallest = i;
        int l = 2 * i, r = 2 * i + 1;
        if (l <= heap -> size && heap -> data[l].cost < heap -> data[smallest].cost)
            smallest = l;
        if (r <= heap -> size && heap -> data[r].cost < heap -> data[smallest].cost)
            smallest = r;
        if (smallest == i) break;
        swap(&heap -> data[i], &heap -> data[smallest]);
        i = smallest;
    }
    return top;
}

bool heapEmpty(MinHeap* heap) {
    return heap -> size == 0;
}

int minimumCost(int n, int** connections, int connectionsSize, int* connectionsColSize) {
    for (int i = 1; i <= n; i++) {
        graph[i] = NULL;
    }

    for (int i = 0; i < connectionsSize; i++) {
        int u = connections[i][0];
        int v = connections[i][1];
        int cost = connections[i][2];
        addEdge(u, v, cost);
    }

    bool visited[MAXN] = { false };
    MinHeap* heap = createHeap(connectionsSize * 2);
    heapPush(heap, 1, 0); 

    int totalCost = 0, count = 0;

    while (!heapEmpty(heap) && count < n) {
        PQNode top = heapPop(heap);
        int node = top.node, cost = top.cost;
        if (visited[node]) continue;
        visited[node] = true;
        totalCost += cost;
        count++;

        for (Edge* e = graph[node]; e; e = e -> next) {
            if (!visited[e -> to]) {
                heapPush(heap, e -> to, e -> cost);
            }
        }
    }

    free(heap -> data);
    free(heap);

    return count == n ? totalCost : -1;
}
