#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

typedef struct {
    int *parent;    
    int *rank;      
    int size;       
} DSU;

DSU* dsu_init(int n) {
    DSU *dsu = malloc(sizeof(DSU));
    dsu -> size = n;
    dsu -> parent = malloc(n * sizeof(int));
    dsu -> rank = malloc(n * sizeof(int));
    for(int i = 0; i < n; i++) {
        dsu -> parent[i] = i;
        dsu -> rank[i] = 0;
    }
    return dsu;
}

int dsu_find(DSU *dsu, int x) {
    if(dsu -> parent[x] != x) {
        dsu -> parent[x] = dsu_find(dsu, dsu -> parent[x]);
    }
    return dsu -> parent[x];
}

bool dsu_union(DSU *dsu, int a, int b) {
    int ra = dsu_find(dsu, a);
    int rb = dsu_find(dsu, b);
    if (ra == rb) return false;
    if (dsu -> rank[ra] < dsu -> rank[rb]) {
        dsu -> parent[ra] = rb;
    } else {
        dsu -> parent[rb] = ra;
        if (dsu -> rank[ra] == dsu -> rank[rb]) {
            dsu -> rank[ra]++;
        }
    }
    return true;
}

void dsu_free(DSU *dsu) {
    free(dsu -> parent);
    free(dsu -> rank);
    free(dsu);
}

typedef struct {
    int u, v;   
    int t;    
    int c;      
} Edge;

typedef struct AdjNode {
    int v;
    int w;
    struct AdjNode *next;
} AdjNode;

typedef struct {
    int v;
    int dist;
} HeapNode;

typedef struct {
    HeapNode *data;
    int size;
} MinHeap;

MinHeap* heap_init(int cap) {
    MinHeap *h = malloc(sizeof(MinHeap));
    h -> data = malloc(cap * sizeof(HeapNode));
    h -> size = 0;
    return h;
}

void heap_swap(HeapNode *a, HeapNode *b) {
    HeapNode tmp = *a;
    *a = *b;
    *b = tmp;
}

void heap_push(MinHeap *h, int v, int dist) {
    int i = h -> size++;
    h -> data[i].v = v;
    h -> data[i].dist = dist;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h -> data[p].dist <= h -> data[i].dist) break;
        heap_swap(&h -> data[p], &h -> data[i]);
        i = p;
    }
}

HeapNode heap_pop(MinHeap *h) {
    HeapNode ret = h -> data[0];
    h -> data[0] = h -> data[--h -> size];
    int i = 0;
    while (true) {
        int l = 2*i + 1, r = 2*i + 2, smallest = i;
        if (l < h -> size && h -> data[l].dist < h -> data[smallest].dist) smallest = l;
        if (r < h -> size && h -> data[r].dist < h -> data[smallest].dist) smallest = r;
        if (smallest == i) break;
        heap_swap(&h -> data[i], &h -> data[smallest]);
        i = smallest;
    }
    return ret;
}

void heap_free(MinHeap *h) {
    free(h -> data);
    free(h);
}

int n, m, s, L;
Edge *edges;
int *comb;
long long best_cost;

bool evaluate_combination(void) {
    long long total_cost = 0;
    for (int i = 0; i < n - 1; i++) total_cost += edges[comb[i]].c;
    if (total_cost >= best_cost) return false;

    DSU *dsu = dsu_init(n);
    for (int i = 0; i < n - 1; i++) {
        Edge *e = &edges[comb[i]];
        if (!dsu_union(dsu, e -> u, e -> v)) {
            dsu_free(dsu);
            return false;
        }
    }
    int root0 = dsu_find(dsu, 0);
    for (int i = 0; i < n; i++) {
        if (dsu_find(dsu, i) != root0) {
            dsu_free(dsu);
            return false;
        }
    }
    dsu_free(dsu);

    AdjNode **adj = calloc(n, sizeof(AdjNode*));
    for (int i = 0; i < n - 1; i++) {
        Edge *e = &edges[comb[i]];
        AdjNode *a = malloc(sizeof(AdjNode)); a -> v = e -> v; a -> w = e -> t;
        a -> next = adj[e -> u]; adj[e -> u] = a;
        AdjNode *b = malloc(sizeof(AdjNode)); b -> v = e -> u; b -> w = e -> t;
        b -> next = adj[e -> v]; adj[e -> v] = b;
    }

    const long long INF = LLONG_MAX;
    long long *dist = malloc(n * sizeof(long long));
    for (int i = 0; i < n; i++) dist[i] = INF;
    dist[s] = 0;
    MinHeap *h = heap_init(m);
    heap_push(h, s, 0);
    while (h -> size > 0) {
        HeapNode cur = heap_pop(h);
        if (cur.dist > dist[cur.v] || cur.dist > L) continue;
        AdjNode *p = adj[cur.v];
        while (p) {
            long long nd = cur.dist + p -> w;
            if (nd < dist[p -> v] && nd <= L) {
                dist[p -> v] = nd;
                heap_push(h, p -> v, nd);
            }
            p = p -> next;
        }
    }
    heap_free(h);

    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (dist[i] == INF) { ok = false; break; }
    }
    free(dist);

    for (int i = 0; i < n; i++) {
        AdjNode *p = adj[i];
        while (p) { AdjNode *tmp = p; p = p -> next; free(tmp); }
    }
    free(adj);

    if (ok) best_cost = total_cost;
    return ok;
}

void gen_combinations(int start, int depth) {
    if (depth == n - 1) {
        evaluate_combination();
        return;
    }
    for (int i = start; i <= m - (n - 1 - depth); i++) {
        comb[depth] = i;
        gen_combinations(i + 1, depth + 1);
    }
}

int main() {
    scanf("%d %d %d %d", &n, &m, &s, &L);
    s -= 1; 

    edges = malloc(m * sizeof(Edge));
    for (int i = 0; i < m; i++) {
        int u,v,t,c;
        scanf("%d %d %d %d", &u, &v, &t, &c);
        edges[i].u = u - 1;
        edges[i].v = v - 1;
        edges[i].t = t;
        edges[i].c = c;
    }

    comb = malloc((n - 1) * sizeof(int));
    best_cost = LLONG_MAX;

    gen_combinations(0, 0);

    if (best_cost == LLONG_MAX) {
        printf("NO_SOLUTION");
    } else {
        printf("%lld", best_cost);
    }

    free(edges);
    free(comb);
    return 0;
}
