#include <stdio.h>
#include <stdint.h>

typedef unsigned __int128 u128;

int n, m;
u128 adj[128];
u128 best_clique = 0;
int max_size = 0;

int popcount128(u128 x) {
    uint64_t lo = (uint64_t)(x);
    uint64_t hi = (uint64_t)(x >> 64);
    return __builtin_popcountll(lo) + __builtin_popcountll(hi);
}

int ctz128(u128 x) {
    uint64_t lo = (uint64_t)(x);
    if (lo) {
        return __builtin_ctzll(lo);
    }
    uint64_t hi = (uint64_t)(x >> 64);
    return 64 + __builtin_ctzll(hi);
}

void bron(u128 R, u128 P, u128 X) {
    if (P == 0 && X == 0) {
        int sz = popcount128(R);
        if (sz > max_size) {
            max_size = sz;
            best_clique = R;
        }
        return;
    }
    u128 PX = P | X;
    int u = ctz128(PX);
    u128 skip = P & ~adj[u];
    while (skip) {
        int v = ctz128(skip);
        skip &= skip - 1;
        u128 vmask = ((u128)1 << v);

        bron(R | vmask, P & adj[v], X & adj[v]);

        P &= ~vmask;
        X |= vmask;
    }
}

int main(){
    scanf("%d %d", &n, &m);
    if (n > 128) return 1;
    for(int i = 0; i < m; i++){
        int u,v;
        scanf("%d %d", &u, &v);
        u -= 1; 
        v -= 1;
        adj[u] |= ((u128)1 << v);
        adj[v] |= ((u128)1 << u);
    }
    u128 P = (n == 128 ? ~((u128)0) : (((u128)1 << n) - 1));
    bron(0, P, 0);

    printf("%d\n", max_size);
    for(int i = 0; i < n; i++){
        if ((best_clique >> i) & 1){
            printf("%d ", i + 1);
        }
    }
    return 0;
}
