#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int n, m;
static int blocks;
static uint64_t *graph;    
static int ans = -1;
static int curr[100], best[100];

static inline int bit_count(const uint64_t *arr) {
    int cnt = 0;
    for (int i = 0; i < blocks; i++) {
        cnt += __builtin_popcountll(arr[i]);
    }
    return cnt;
}

void backtrack(int size, uint64_t *cands) {
    int cnt = bit_count(cands);
    if (size + cnt <= ans) return;

    for (int b = 0; b < blocks; b++) {
        uint64_t w = cands[b];
        while (w) {
            int bit = __builtin_ctzll(w);
            int v = b * 64 + bit;
            w &= w - 1;

            curr[size] = v;
            if (size + 1 > ans) {
                ans = size + 1;
                memcpy(best, curr, ans * sizeof(int));
            }

            uint64_t newCand[blocks];
            for (int i = 0; i < blocks; i++) {
                newCand[i] = cands[i] & graph[(uint64_t)v * blocks + i];
            }

            backtrack(size + 1, newCand);

            cands[b] &= ~(1ULL << bit);
        }
    }
}

int main() {
    scanf("%d %d", &n, &m);
    blocks = (n + 63) / 64;
    graph = calloc(n * blocks, sizeof(uint64_t));
    
    for (int i = 0; i < m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        u -= 1; 
        v -= 1;
        graph[u * blocks + (v / 64)] |= (1ULL << (v % 64));
        graph[v * blocks + (u / 64)] |= (1ULL << (u % 64));
    }

    uint64_t cands[blocks];
    for (int i = 0; i < blocks; i++) {
        cands[i] = (i < (n + 63) / 64 - 1) ? UINT64_MAX: ((n % 64) ? ((1ULL << (n % 64)) - 1) : UINT64_MAX);
    }

    backtrack(0, cands);

    printf("%d\n", ans);
    for (int i = 0; i < ans; i++) {
        printf("%d ", best[i] + 1);
    }

    free(graph);
    return 0;
}
