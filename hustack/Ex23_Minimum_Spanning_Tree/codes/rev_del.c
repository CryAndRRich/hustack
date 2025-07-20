#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAXN 100005
#define MAXM 100005

typedef struct {
    int w, u, v;
} Edge;

typedef struct Node {
    int v;
    struct Node* next;
} Node;

int n, m;
Edge edges[MAXM];
Node* adj[MAXN];
bool visited[MAXN];

int cmp(const void* a, const void* b) {
    return ((Edge*)b) -> w - ((Edge*)a) -> w;  
}

void addEdge(int u, int v) {
    Node* node = (Node*)malloc(sizeof(Node));
    node -> v = v;
    node -> next = adj[u];
    adj[u] = node;
}

void removeEdge(int u, int v) {
    Node **cur = &adj[u];
    while (*cur) {
        if ((*cur) -> v == v) {
            Node* tmp = *cur;
            *cur = (*cur) -> next;
            free(tmp);
            return;
        }
        cur = &(*cur) -> next;
    }
}

bool hasEdge(int u, int v) {
    Node* cur = adj[u];
    while (cur) {
        if (cur -> v == v) return true;
        cur = cur -> next;
    }
    return false;
}

void dfs(int u) {
    visited[u] = true;
    for (Node* cur = adj[u]; cur; cur = cur -> next) {
        if (!visited[cur -> v]) {
            dfs(cur -> v);
        }
    }
}

void resetVisited() {
    for (int i = 1; i <= n; ++i)
        visited[i] = false;
}

int main() {
    scanf("%d %d", &n, &m);

    for (int i = 0; i < m; ++i) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        edges[i] = (Edge){w, u, v};
        addEdge(u, v);
        addEdge(v, u);
    }

    qsort(edges, m, sizeof(Edge), cmp);

    int total = 0;

    for (int i = 0; i < m; ++i) {
        int u = edges[i].u;
        int v = edges[i].v;
        int w = edges[i].w;

        if (!hasEdge(u, v)) continue;

        removeEdge(u, v);
        removeEdge(v, u);

        resetVisited();
        dfs(1);

        bool connected = true;
        for (int j = 1; j <= n; ++j) {
            if (!visited[j]) {
                connected = false;
                break;
            }
        }

        if (!connected) {
            addEdge(u, v);
            addEdge(v, u);
            total += w;
        }
    }

    printf("%d", total);
    return 0;
}
