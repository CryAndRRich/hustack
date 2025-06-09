#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXN 100005
#define INF 1000000000

typedef struct {
    int v, w;
} Edge;

typedef struct {
    Edge *edges;
    int size, capacity;
} Node;

Node graph[MAXN];
int visited[MAXN];

void add_edge(int u, int v, int w) {
    if (graph[u].size == graph[u].capacity) {
        graph[u].capacity = graph[u].capacity ? graph[u].capacity * 2 : 2;
        graph[u].edges = realloc(graph[u].edges, graph[u].capacity * sizeof(Edge));
    }
    graph[u].edges[graph[u].size++] = (Edge){v, w};
}

typedef struct {
    int node, cost;
} PQNode;

PQNode heap[MAXN];
int heap_size = 0;

void swap(PQNode *a, PQNode *b) {
    PQNode tmp = *a; *a = *b; *b = tmp;
}

void push(PQNode x) {
    heap[++heap_size] = x;
    int i = heap_size;
    while (i > 1 && heap[i].cost < heap[i/2].cost) {
        swap(&heap[i], &heap[i/2]);
        i /= 2;
    }
}

PQNode pop() {
    PQNode top = heap[1];
    heap[1] = heap[heap_size--];
    int i = 1;
    while (2 * i <= heap_size) {
        int j = 2 * i;
        if (j < heap_size && heap[j+1].cost < heap[j].cost) j++;
        if (heap[i].cost <= heap[j].cost) break;
        swap(&heap[i], &heap[j]);
        i = j;
    }
    return top;
}

int prim(int start) {
    memset(visited, 0, sizeof(visited));
    heap_size = 0;
    push((PQNode){start, 0});
    int total = 0;

    while (heap_size) {
        PQNode curr = pop();
        int u = curr.node;
        if (visited[u]) continue;
        visited[u] = 1;
        total += curr.cost;

        for (int i = 0; i < graph[u].size; ++i) {
            int v = graph[u].edges[i].v;
            int w = graph[u].edges[i].w;
            if (!visited[v]) {
                push((PQNode){v, w});
            }
        }
    }
    return total;
}

int main() {
    int n, m; scanf("%d %d", &n, &m);
    for (int i = 0; i < m; ++i) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        add_edge(u, v, w);
        add_edge(v, u, w);
    }

    int res = prim(1);
    printf("%d", res);
    return 0;
}
