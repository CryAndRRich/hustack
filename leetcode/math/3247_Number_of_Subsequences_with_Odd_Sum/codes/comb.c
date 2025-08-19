#include <stdio.h>
#define MOD 1000000007
#define MAXN 100005

long long powmod(long long base, long long exp) {
    long long result = 1;
    while (exp > 0) {
        if (exp % 2) result = result * base % MOD;
        base = base * base % MOD;
        exp /= 2;
    }
    return result;
}

long long fact[MAXN], invfact[MAXN];

void init_factorials() {
    fact[0] = 1;
    for (int i = 1; i < MAXN; i++) {
        fact[i] = fact[i - 1] * i % MOD;
    }
    invfact[MAXN - 1] = powmod(fact[MAXN - 1], MOD - 2);
    for (int i = MAXN - 2; i >= 0; i--) {
        invfact[i] = invfact[i + 1] * (i + 1) % MOD;
    }
}

long long comb(int n, int k) {
    if (k < 0 || k > n) return 0;
    return fact[n] * invfact[k] % MOD * invfact[n - k] % MOD;
}

int subsequenceCount(int* nums, int numsSize) {
    int even = 0, odd = 0;
    for (int i = 0; i < numsSize; i++) {
        if (nums[i] % 2 == 0) even++;
        else odd++;
    }

    init_factorials();

    long long res = 0;
    for (int k = 1; k <= odd; k += 2) {
        long long ways = comb(odd, k) * powmod(2, even) % MOD;
        res = (res + ways) % MOD;
    }

    return (int)res;
}
