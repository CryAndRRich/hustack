#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
    int to;
    int w;
    struct Node* next;
} Node;

typedef struct Query {
    int v;
    int idx;
    struct Query* next;
} Query;

typedef struct {
    int *parent;
    int *rank;
} DSU;

DSU* createDSU(int n) {
    DSU *dsu = (DSU*)malloc(sizeof(DSU));
    dsu -> parent = (int*)malloc(sizeof(int) * n);
    dsu -> rank = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        dsu -> parent[i] = i;
        dsu -> rank[i] = 0;
    }
    return dsu;
}

int findDSU(DSU* dsu, int x) {
    int r = x;
    while (dsu -> parent[r] != r) r = dsu -> parent[r];
    while (dsu -> parent[x] != x) { int nxt = dsu -> parent[x]; dsu -> parent[x] = r; x = nxt; }
    return r;
}

void unionDSU(DSU* dsu, int x, int y) {
    int rx = findDSU(dsu, x);
    int ry = findDSU(dsu, y);
    if (rx == ry) return;
    if (dsu -> rank[rx] < dsu -> rank[ry]) {
        dsu -> parent[rx] = ry;
    } else if (dsu -> rank[rx] > dsu -> rank[ry]) {
        dsu -> parent[ry] = rx;
    } else {
        dsu -> parent[ry] = rx;
        dsu -> rank[rx]++;
    }
}

void addEdge(Node **adj, int u, int v, int w) {
    Node *node = (Node*)malloc(sizeof(Node));
    node -> to = v;
    node -> w = w;
    node -> next = adj[u];
    adj[u] = node;
}

void addQuery(Query **queries, int u, int v, int idx) {
    Query *q = (Query*)malloc(sizeof(Query));
    q -> v = v;
    q -> idx = idx;
    q -> next = queries[u];
    queries[u] = q;
}

void tarjanDFS(int u, int p, Node **adj, Query **queries, int *visited, int **freq,
               DSU *dsu, int *ancestor, int *result) {
    ancestor[u] = u;
    for (Node *e = adj[u]; e != NULL; e = e -> next) {
        int v = e -> to;
        if (v == p) continue;
        for (int k = 1; k <= 26; k++) freq[v][k] = freq[u][k];
        freq[v][e -> w]++;
        tarjanDFS(v, u, adj, queries, visited, freq, dsu, ancestor, result);
        unionDSU(dsu, v, u);
        ancestor[findDSU(dsu, u)] = u;
    }
    visited[u] = 1;
    for (Query *q = queries[u]; q != NULL; q = q -> next) {
        int v = q -> v;
        if (visited[v]) {
            int lca = ancestor[findDSU(dsu, v)];
            int sum = 0, mx = 0;
            for (int k = 1; k <= 26; k++) {
                int c = freq[u][k] + freq[v][k] - 2 * freq[lca][k];
                sum += c;
                if (c > mx) mx = c;
            }
            result[q -> idx] = sum - mx;
        }
    }
}

int* minOperationsQueries(int n, int** edges, int edgesSize, int* edgesColSize,
                          int** queries, int queriesSize, int* queriesColSize, int* returnSize) {
    *returnSize = queriesSize;
    int *result = (int*)malloc(sizeof(int) * queriesSize);
    for (int i = 0; i < queriesSize; ++i) result[i] = 0;

    if (n <= 1) {
        for (int i = 0; i < queriesSize; ++i) result[i] = 0;
        return result;
    }

    Node **adj = (Node**)calloc(n, sizeof(Node*));
    for (int i = 0; i < edgesSize; i++) {
        int u = edges[i][0], v = edges[i][1], w = edges[i][2];
        addEdge(adj, u, v, w);
        addEdge(adj, v, u, w);
    }

    Query **qList = (Query**)calloc(n, sizeof(Query*));
    for (int i = 0; i < queriesSize; i++) {
        int u = queries[i][0], v = queries[i][1];
        addQuery(qList, u, v, i);
        addQuery(qList, v, u, i);
    }

    int **freq = (int**)malloc(sizeof(int*) * n);
    for (int i = 0; i < n; i++) {
        freq[i] = (int*)calloc(27, sizeof(int));
    }

    int *visited = (int*)calloc(n, sizeof(int));
    int *ancestor = (int*)malloc(sizeof(int) * n);

    DSU *dsu = createDSU(n);

    tarjanDFS(0, -1, adj, qList, visited, freq, dsu, ancestor, result);

    for (int i = 0; i < n; i++) {
        Node *cur = adj[i];
        while (cur) {
            Node *tmp = cur;
            cur = cur -> next;
            free(tmp);
        }
    }
    free(adj);

    for (int i = 0; i < n; i++) {
        Query *cur = qList[i];
        while (cur) {
            Query *tmp = cur;
            cur = cur -> next;
            free(tmp);
        }
    }
    free(qList);

    for (int i = 0; i < n; i++) free(freq[i]);
    free(freq);
    free(visited);
    free(ancestor);
    free(dsu -> parent);
    free(dsu -> rank);
    free(dsu);

    return result;
}
