#include <stdio.h>
#include <stdlib.h>

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

void dfs(int node, int parent, int dist, AdjList* graph, int *maxDist, int *farthestNode) {
    if (dist > *maxDist) {
        *maxDist = dist;
        *farthestNode = node;
    }
    for (int i = 0; i < graph[node].size; i++) {
        int nxt = graph[node].neighbors[i];
        if (nxt != parent) {
            dfs(nxt, node, dist + 1, graph, maxDist, farthestNode);
        }
    }
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
    
    int maxDist = -1, farthestNode = 0;
    dfs(0, -1, 0, graph, &maxDist, &farthestNode);
    
    maxDist = -1;
    dfs(farthestNode, -1, 0, graph, &maxDist, &farthestNode);
    
    for (int i = 0; i < n; i++) free(graph[i].neighbors);
    free(graph);
    
    return maxDist;
}
