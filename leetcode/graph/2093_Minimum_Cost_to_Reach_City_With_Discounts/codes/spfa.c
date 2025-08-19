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
    int capacity;
    int n;
} Graph;

void initGraph(Graph* g, int n) {
    g -> n = n;
    g -> capacity = 2000;  
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
    int node;
    int discounts_left;
} State;

typedef struct {
    State* data;
    int front, rear, capacity;
} Queue;

Queue* createQueue(int capacity) {
    Queue* q = (Queue*)malloc(sizeof(Queue));
    q -> data = (State*)malloc(sizeof(State) * capacity);
    q -> front = q -> rear = 0;
    q -> capacity = capacity;
    return q;
}

int isEmpty(Queue* q) {
    return q -> front == q -> rear;
}

void enqueue(Queue* q, State s) {
    q -> data[q -> rear++] = s;
    if (q -> rear == q -> capacity) q -> rear = 0;
}

State dequeue(Queue* q) {
    State s = q -> data[q -> front++];
    if (q -> front == q -> capacity) q -> front = 0;
    return s;
}

int minimumCost(int n, int** highways, int highwaysSize, int* highwaysColSize, int discounts) {
    Graph g;
    initGraph(&g, n);

    for (int i = 0; i < highwaysSize; i++) {
        int u = highways[i][0];
        int v = highways[i][1];
        int cost = highways[i][2];
        addEdge(&g, u, v, cost);
        addEdge(&g, v, u, cost);
    }

    int **dist = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++) {
        dist[i] = (int*)malloc(sizeof(int) * (discounts + 1));
        for (int j = 0; j <= discounts; j++) dist[i][j] = INT_MAX;
    }

    int **in_queue = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++) {
        in_queue[i] = (int*)calloc(discounts + 1, sizeof(int));
    }

    dist[0][discounts] = 0;
    Queue* q = createQueue(n * (discounts + 1));
    enqueue(q, (State){0, discounts});
    in_queue[0][discounts] = 1;

    while (!isEmpty(q)) {
        State cur = dequeue(q);
        int u = cur.node, d_left = cur.discounts_left;
        in_queue[u][d_left] = 0;
        int curDist = dist[u][d_left];

        for (int i = g.head[u]; i != -1; i = g.edges[i].next) {
            int v = g.edges[i].to;
            int c = g.edges[i].cost;

            if (curDist != INT_MAX && curDist + c < dist[v][d_left]) {
                dist[v][d_left] = curDist + c;
                if (!in_queue[v][d_left]) {
                    enqueue(q, (State){v, d_left});
                    in_queue[v][d_left] = 1;
                }
            }

            if (d_left > 0) {
                int cDisc = curDist + c / 2;
                if (curDist != INT_MAX && cDisc < dist[v][d_left - 1]) {
                    dist[v][d_left - 1] = cDisc;
                    if (!in_queue[v][d_left - 1]) {
                        enqueue(q, (State){v, d_left - 1});
                        in_queue[v][d_left - 1] = 1;
                    }
                }
            }
        }
    }

    int ans = INT_MAX;
    for (int i = 0; i <= discounts; i++) {
        if (dist[n - 1][i] < ans) ans = dist[n - 1][i];
    }

    for (int i = 0; i < n; i++) {
        free(dist[i]);
        free(in_queue[i]);
    }
    free(dist);
    free(in_queue);
    free(g.edges);
    free(g.head);
    free(q -> data);
    free(q);

    return ans == INT_MAX ? -1 : ans;
}
