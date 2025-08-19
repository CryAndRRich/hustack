#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
typedef struct {
    int v, w; 
    struct Node* next;
} Edge;

typedef struct Node {
    int v, w; 
    struct Node* next;
} Node;

typedef struct {
    long long d; 
    int u;} 
PQItem;

typedef struct {
    PQItem *a; 
    int n, c;} 
MinHeap;

void push(MinHeap* h, PQItem it) {
    if (h -> n == h -> c) {
        h -> c = h -> c ? h -> c*2 : 4; 
        h -> a = realloc(h -> a, h -> c*sizeof(PQItem));
    }
    h -> a[h -> n++] = it;
    for (int i = h -> n-1; i > 0; ) {
        int p = (i-1)/2;
        if (h -> a[p].d <= h -> a[i].d) break;
        PQItem t = h -> a[p]; h -> a[p] = h -> a[i]; h -> a[i] = t;
        i = p;
    }
}
PQItem pop(MinHeap* h) {
    PQItem r = h -> a[0];
    h -> a[0] = h -> a[--h -> n];
    for (int i = 0; ; ) {
        int l = i*2+1, rgt = l+1, m = i;
        if (l < h -> n && h -> a[l].d < h -> a[m].d) m = l;
        if (rgt < h -> n && h -> a[rgt].d < h -> a[m].d) m = rgt;
        if (m == i) break;
        PQItem t = h -> a[m]; h -> a[m] = h -> a[i]; h -> a[i] = t;
        i = m;
    }
    return r;
}
long long* minCost(int n, int** roads, int roadsSize, int* roadsColSize, int* appleCost, int appleCostSize, int k, int* returnSize) {
    Node** g = calloc(n, sizeof(Node*));
    for (int i = 0; i < roadsSize; i++) {
        int a = roads[i][0]-1, b = roads[i][1]-1, c = roads[i][2];
        Node* e1 = malloc(sizeof(Node)); e1 -> v = b; e1 -> w = c; e1 -> next = g[a]; g[a] = e1;
        Node* e2 = malloc(sizeof(Node)); e2 -> v = a; e2 -> w = c; e2 -> next = g[b]; g[b] = e2;
    }
    long long* ans = malloc(sizeof(long long)*n);
    for (int s = 0; s < n; s++) {
        long long* dist = malloc(sizeof(long long)*n);
        for (int i = 0; i < n; i++) dist[i] = LLONG_MAX;
        dist[s] = 0;
        MinHeap pq = {NULL,0,0};
        push(&pq, (PQItem){0,s});
        long long res = LLONG_MAX;
        while (pq.n) {
            PQItem it = pop(&pq);
            long long d = it.d;
            int u = it.u;
            if (d > dist[u]) continue;
            long long total = (long long)appleCost[u] + (long long)(k+1)*d;
            if (total < res) res = total;
            for (Node* e = g[u]; e; e = e -> next) {
                long long nd = d + e -> w;
                if (nd < dist[e -> v]) {
                    dist[e -> v] = nd;
                    push(&pq, (PQItem){nd, e -> v});
                }
            }
        }
        ans[s] = res;
        free(dist);
        free(pq.a);
    }
    *returnSize = n;
    return ans;
}
