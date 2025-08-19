#include <stdio.h>
#include <stdlib.h>

#define MOD 1000000007LL

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

void build_fact_and_ifact(int n, long long *fact, long long *ifact) {
    fact[0] = 1;
    for (int i = 1; i <= n; i++) fact[i] = fact[i - 1] * i % MOD;
    ifact[n] = modpow(fact[n], MOD - 2);
    for (int i = n; i > 0; i--) ifact[i - 1] = ifact[i] * i % MOD;
}

long long nCk_mod(int n, int k, long long *fact, long long *ifact) {
    if (k < 0 || k > n) return 0;
    return fact[n] * ifact[k] % MOD * ifact[n - k] % MOD;
}

long long stirling2_mod(int n, int k) {
    if (k < 0 || k > n) return 0;
    long long *fact = malloc((k + 1) * sizeof(long long));
    long long *ifact = malloc((k + 1) * sizeof(long long));
    build_fact_and_ifact(k, fact, ifact);
    long long fact_k = fact[k];
    long long total = 0;
    for (int m = 0; m <= k; m++) {
        long long binom = nCk_mod(k, m, fact, ifact);
        long long term = binom * modpow(k - m, n) % MOD;
        if (m & 1) total = (total - term + MOD) % MOD;
        else total = (total + term) % MOD;
    }
    long long inv_fact_k = modpow(fact_k, MOD - 2);
    free(fact);
    free(ifact);
    return total * inv_fact_k % MOD;
}

struct Solution {
    int (*waysToDistribute)(int, int);
};

int waysToDistribute_impl(int n, int k) {
    return (int)stirling2_mod(n, k);
}

struct Solution newSolution() {
    struct Solution sol;
    sol.waysToDistribute = waysToDistribute_impl;
    return sol;
}
