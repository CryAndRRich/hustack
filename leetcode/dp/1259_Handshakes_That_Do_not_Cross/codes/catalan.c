#include <stdio.h>

#define MOD 1000000007
#define MAXN 1001

long long fact[2 * MAXN];

long long mod_pow(long long a, long long b, int mod) {
    long long res = 1;
    while (b > 0) {
        if (b % 2) res = res * a % mod;
        a = a * a % mod;
        b /= 2;
    }
    return res;
}

void compute_factorials(int n) {
    fact[0] = 1;
    for (int i = 1; i <= 2 * n; i++) {
        fact[i] = fact[i - 1] * i % MOD;
    }
}

int numberOfWays(int numPeople) {
    int n = numPeople / 2;
    compute_factorials(n);

    long long numerator = fact[2 * n];
    long long denom = (fact[n + 1] * fact[n]) % MOD;
    long long denom_inv = mod_pow(denom, MOD - 2, MOD);

    long long catalan = (numerator * denom_inv) % MOD;
    return (int)catalan;
}
