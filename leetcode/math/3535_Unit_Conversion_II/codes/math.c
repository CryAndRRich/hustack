#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOD 1000000007

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e) {
        if (e & 1) r = (r * a) % MOD;
        a = (a * a) % MOD;
        e >>= 1;
    }
    return r;
}

typedef struct { int v; long long f; int forward; } Edge;

int* queryConversions(int** conversions, int conversionsSize, int* conversionsColSize, int** queries, int queriesSize, int* queriesColSize, int* returnSize) {
    int n = conversionsSize + 1;
    Edge** g = malloc(n * sizeof(Edge*));
    int* deg = calloc(n, sizeof(int));
    for (int i = 0; i < conversionsSize; i++) {
        int a = conversions[i][0], b = conversions[i][1], f = conversions[i][2];
        deg[a]++; deg[b]++;
    }
    for (int i = 0; i < n; i++) g[i] = malloc(deg[i] * sizeof(Edge)), deg[i] = 0;
    for (int i = 0; i < conversionsSize; i++) {
        int a = conversions[i][0], b = conversions[i][1], f = conversions[i][2];
        g[a][deg[a]++] = (Edge){b, f, 1};
        g[b][deg[b]++] = (Edge){a, f, 0};
    }
    long long* num = calloc(n, sizeof(long long));
    long long* den = calloc(n, sizeof(long long));
    int* vis = calloc(n, sizeof(int));
    int* q = malloc(n * sizeof(int));
    int front = 0, back = 0;
    num[0] = 1; den[0] = 1; vis[0] = 1; q[back++] = 0;
    while (front < back) {
        int u = q[front++];
        for (int i = 0; i < deg[u]; i++) {
            int v = g[u][i].v;
            long long f = g[u][i].f;
            int forward = g[u][i].forward;
            if (!vis[v]) {
                if (forward) {
                    num[v] = num[u];
                    den[v] = (den[u] * f) % MOD;
                } else {
                    num[v] = (num[u] * f) % MOD;
                    den[v] = den[u];
                }
                vis[v] = 1;
                q[back++] = v;
            }
        }
    }
    int* ans = malloc(queriesSize * sizeof(int));
    for (int i = 0; i < queriesSize; i++) {
        int a = queries[i][0], b = queries[i][1];
        long long pa = num[a], qa = den[a];
        long long pb = num[b], qb = den[b];
        long long p = (pa * qb) % MOD;
        long long qv = (qa * pb) % MOD;
        long long inv_q = modpow(qv, MOD - 2);
        ans[i] = (int)((p * inv_q) % MOD);
    }
    *returnSize = queriesSize;
    return ans;
}
