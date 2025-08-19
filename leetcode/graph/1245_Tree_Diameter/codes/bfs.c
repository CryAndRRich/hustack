#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int *data;
    int front, rear, capacity;
} Queue;

Queue* createQueue(int capacity) {
    Queue* q = (Queue*)malloc(sizeof(Queue));
    q->data = (int*)malloc(sizeof(int) * capacity);
    q->front = q->rear = 0;
    q->capacity = capacity;
    return q;
}

int isEmpty(Queue* q) {
    return q->front == q->rear;
}

void enqueue(Queue* q, int val) {
    q->data[q->rear++] = val;
}

int dequeue(Queue* q) {
    return q->data[q->front++];
}

void freeQueue(Queue* q) {
    free(q->data);
    free(q);
}

typedef struct {
    int *neighbors;
    int size;
    int capacity;
} AdjList;

void addEdge(AdjList* graph, int u, int v) {
    if (graph[u].size == graph[u].capacity) {
        graph[u].capacity *= 2;
        graph[u].neighbors = (int*)realloc(graph[u].neighbors, sizeof(int)*graph[u].capacity);
    }
    graph[u].neighbors[graph[u].size++] = v;
}

void bfs(int start, AdjList* graph, int n, int *farthest_node, int *max_dist) {
    int *visited = (int*)calloc(n, sizeof(int));
    int *dist = (int*)calloc(n, sizeof(int));
    Queue *q = createQueue(n);
    
    enqueue(q, start);
    visited[start] = 1;
    dist[start] = 0;
    *farthest_node = start;
    *max_dist = 0;
    
    while (!isEmpty(q)) {
        int u = dequeue(q);
        for (int i = 0; i < graph[u].size; i++) {
            int v = graph[u].neighbors[i];
            if (!visited[v]) {
                visited[v] = 1;
                dist[v] = dist[u] + 1;
                enqueue(q, v);
                if (dist[v] > *max_dist) {
                    *max_dist = dist[v];
                    *farthest_node = v;
                }
            }
        }
    }
    
    free(visited);
    free(dist);
    freeQueue(q);
}

int treeDiameter(int** edges, int edgesSize, int* edgesColSize){
    int n = edgesSize + 1;
    AdjList* graph = (AdjList*)malloc(sizeof(AdjList)*n);
    for (int i = 0; i < n; i++) {
        graph[i].neighbors = (int*)malloc(sizeof(int)*2);
        graph[i].size = 0;
        graph[i].capacity = 2;
    }
    for (int i = 0; i < edgesSize; i++) {
        int u = edges[i][0], v = edges[i][1];
        addEdge(graph, u, v);
        addEdge(graph, v, u);
    }
    
    int nodeB, maxDist;
    bfs(0, graph, n, &nodeB, &maxDist);
    bfs(nodeB, graph, n, &nodeB, &maxDist);
    
    for (int i = 0; i < n; i++) free(graph[i].neighbors);
    free(graph);
    
    return maxDist;
}
