#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>

#define INF 1000000000

typedef struct {
    int n, *matchX, *matchY, *prev;
    bool *visX, *visY;
    int *lx, *ly, **g, *slack;
} Hungarian;

Hungarian* createHungarian(int _n, int _m) {
    int n = _n > _m ? _n : _m;
    Hungarian* h = (Hungarian*)malloc(sizeof(Hungarian));
    h -> n = n;
    h -> matchX = (int*)malloc(n * sizeof(int));
    h -> matchY = (int*)malloc(n * sizeof(int));
    h -> prev = (int*)malloc(n * sizeof(int));
    h -> visX = (bool*)malloc(n * sizeof(bool));
    h -> visY = (bool*)malloc(n * sizeof(bool));
    h -> lx = (int*)malloc(n * sizeof(int));
    h -> ly = (int*)malloc(n * sizeof(int));
    h -> slack = (int*)malloc(n * sizeof(int));

    h -> g = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        h -> g[i] = (int*)malloc(n * sizeof(int));
        memset(h -> g[i], 0, n * sizeof(int));
    }

    memset(h -> matchX, -1, n * sizeof(int));
    memset(h -> matchY, -1, n * sizeof(int));
    memset(h -> lx, 0, n * sizeof(int));
    memset(h -> ly, 0, n * sizeof(int));

    return h;
}

void addEdge(Hungarian* h, int u, int v, int w) {
    h -> g[u][v] = w > 0 ? w : 0;
}

bool check(Hungarian* h, int v, int* q, int* front, int* rear) {
    h -> visY[v] = true;
    if (h -> matchY[v] != -1) {
        q[(*rear)++] = h -> matchY[v];
        h -> visX[h -> matchY[v]] = true;
        return false;
    }
    while (v != -1) {
        h -> matchY[v] = h -> prev[v];
        int temp = v;
        v = h -> matchX[h -> prev[temp]];
        h -> matchX[h -> prev[temp]] = temp;
    }
    return true;
}

void bfs(Hungarian* h, int i) {
    int q[h -> n], front = 0, rear = 0;
    memset(h -> visX, 0, h -> n * sizeof(bool));
    memset(h -> visY, 0, h -> n * sizeof(bool));
    for (int j = 0; j < h -> n; j++) h -> slack[j] = INF;

    q[rear++] = i;
    h -> visX[i] = true;

    while (1) {
        while (front < rear) {
            int u = q[front++];
            for (int v = 0; v < h -> n; v++) {
                if (!h -> visY[v]) {
                    int delta = h -> lx[u] + h -> ly[v] - h -> g[u][v];
                    if (h -> slack[v] >= delta) {
                        h -> prev[v] = u;
                        if (delta) {
                            h -> slack[v] = delta;
                        } else if (check(h, v, q, &front, &rear)) {
                            return;
                        }
                    }
                }
            }
        }

        int a = INF;
        for (int j = 0; j < h -> n; j++)
            if (!h -> visY[j] && h -> slack[j] < a)
                a = h -> slack[j];

        for (int j = 0; j < h -> n; j++) {
            if (h -> visX[j]) h -> lx[j] -= a;
            if (h -> visY[j]) h -> ly[j] += a;
            else h -> slack[j] -= a;
        }

        for (int j = 0; j < h -> n; j++)
            if (!h -> visY[j] && h -> slack[j] == 0 && check(h, j, q, &front, &rear))
                return;
    }
}

int findMinimumTime(int* strength, int n) {
    Hungarian* h = createHungarian(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            addEdge(h, i, j, 1000000 - (strength[i] + j) / (j + 1));

    for (int i = 0; i < h -> n; i++)
        h -> lx[i] = 0;

    for (int i = 0; i < h -> n; i++)
        for (int j = 0; j < h -> n; j++)
            if (h -> g[i][j] > h -> lx[i])
                h -> lx[i] = h -> g[i][j];

    for (int i = 0; i < h -> n; i++)
        bfs(h, i);

    int res = 0;
    for (int i = 0; i < h -> n; i++)
        if (h -> matchX[i] != -1)
            res += h -> g[i][h -> matchX[i]];

    res = n * 1000000 - res;

    free(h -> matchX);
    free(h -> matchY);
    free(h -> prev);
    free(h -> visX);
    free(h -> visY);
    free(h -> lx);
    free(h -> ly);
    free(h -> slack);
    for (int i = 0; i < h -> n; i++)
        free(h -> g[i]);
    free(h -> g);
    free(h);

    return res;
}
