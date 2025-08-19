#include <stdio.h>
#include <stdlib.h>

#define MOD 1000000007
#define MAX 1000005

long long fact[MAX], inv_fact[MAX];

long long power(long long a, long long b) {
    long long result = 1;
    a %= MOD;
    while (b > 0) {
        if (b & 1)
            result = result * a % MOD;
        a = a * a % MOD;
        b >>= 1;
    }
    return result;
}

void preprocess() {
    fact[0] = inv_fact[0] = 1;
    for (int i = 1; i < MAX; ++i) {
        fact[i] = fact[i - 1] * i % MOD;
    }
    inv_fact[MAX - 1] = power(fact[MAX - 1], MOD - 2);
    for (int i = MAX - 2; i >= 1; --i) {
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD;
    }
}

int findDerangement(int n) {
    if (n == 1) return 0;
    preprocess();

    long long sum = 0;
    int sign = 1;
    for (int k = 0; k <= n; ++k) {
        long long term = inv_fact[k];
        if (sign == 1)
            sum = (sum + term) % MOD;
        else
            sum = (sum - term + MOD) % MOD;
        sign *= -1;
    }

    long long result = fact[n] * sum % MOD;
    return (int)result;
}
