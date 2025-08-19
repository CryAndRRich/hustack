#include <stdlib.h>
#include <limits.h>

static long long dfs(int mask, int n, int* p, long long* memo) {
    if (mask == (1 << n) - 1) return 0;
    if (memo[mask] != -1) return memo[mask];
    int g = 1 + __builtin_popcount((unsigned)mask);
    long long best = LLONG_MAX;
    for (int i = 0; i < n; i++) {
        if (!(mask & (1 << i))) {
            long long days = (p[i] + (long long)g - 1) / (long long)g;
            long long v = days + dfs(mask | (1 << i), n, p, memo);
            if (v < best) best = v;
        }
    }
    memo[mask] = best;
    return best;
}

long long minimumTime(int* power, int powerSize) {
    int size = 1 << powerSize;
    long long* memo = (long long*)malloc((size_t)size * sizeof(long long));
    for (int i = 0; i < size; i++) memo[i] = -1;
    long long ans = dfs(0, powerSize, power, memo);
    free(memo);
    return ans;
}
