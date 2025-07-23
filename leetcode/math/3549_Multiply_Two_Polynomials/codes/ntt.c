#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOD 998244353
#define PRIMITIVE_ROOT 3

int mod_pow(int a, int b, int mod) {
    int res = 1;
    while (b) {
        if (b & 1) res = (long long)res * a % mod;
        a = (long long)a * a % mod;
        b >>= 1;
    }
    return res;
}

void bit_reverse(int* a, int n) {
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            int tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }
}

void ntt(int* a, int n, int invert) {
    bit_reverse(a, n);
    for (int len = 2; len <= n; len <<= 1) {
        int wlen = mod_pow(PRIMITIVE_ROOT, (MOD - 1) / len, MOD);
        if (invert)
            wlen = mod_pow(wlen, MOD - 2, MOD);
        for (int i = 0; i < n; i += len) {
            int w = 1;
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j];
                int v = (long long)a[i + j + len / 2] * w % MOD;
                a[i + j] = (u + v) % MOD;
                a[i + j + len / 2] = (u - v + MOD) % MOD;
                w = (long long)w * wlen % MOD;
            }
        }
    }
    if (invert) {
        int inv_n = mod_pow(n, MOD - 2, MOD);
        for (int i = 0; i < n; ++i)
            a[i] = (long long)a[i] * inv_n % MOD;
    }
}

long long* multiply(int* poly1, int poly1Size, int* poly2, int poly2Size, int* returnSize) {
    int n = 1;
    while (n < poly1Size + poly2Size - 1) n <<= 1;
    int* a = (int*)calloc(n, sizeof(int));
    int* b = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < poly1Size; ++i) a[i] = (poly1[i] % MOD + MOD) % MOD;
    for (int i = 0; i < poly2Size; ++i) b[i] = (poly2[i] % MOD + MOD) % MOD;

    ntt(a, n, 0);
    ntt(b, n, 0);
    for (int i = 0; i < n; ++i)
        a[i] = (long long)a[i] * b[i] % MOD;
    ntt(a, n, 1);

    *returnSize = poly1Size + poly2Size - 1;
    long long* result = (long long*)malloc(sizeof(long long) * (*returnSize));
    for (int i = 0; i < *returnSize; ++i)
        result[i] = a[i];

    free(a);
    free(b);
    return result;
}
