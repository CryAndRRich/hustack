#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define MOD 1000000000LL
#define MAX 1000

int* sieve(int n, int* count) {
    bool mark[n+1];
    for (int i = 0; i <= n; i++) {
        mark[i] = false;
    }
    for (int p = 2; p * p <= n; p++) {
        if (!mark[p]) {
            for (int i = p * p; i <= n; i += p) {
                mark[i] = true;
            }
        }
    }
    int primeCount = 0;
    for (int i = 2; i <= n; i++) {
        if (!mark[i])
            primeCount++;
    }
    *count = primeCount;
    int* primes = (int*) malloc(primeCount * sizeof(int));
    int index = 0;
    for (int i = 2; i <= n; i++) {
        if (!mark[i])
            primes[index++] = i;
    }
    return primes;
}

long long modExp(long long base, int exp) {
    long long result = 1;
    while (exp > 0) {
        if (exp & 1)
            result = (result * base) % MOD;
        base = (base * base) % MOD;
        exp /= 2;
    }
    return result;
}

long long lcm(const int *nums, int n) {
    int primeCount;
    int* primes = sieve(MAX, &primeCount);
    
    int maxExp[1001] = {0};
    for (int i = 0; i < n; i++) {
        int temp = nums[i];
        for (int j = 0; j < primeCount && temp > 1; j++) {
            int p = primes[j];
            if (p * p > temp) {
                if (temp > 1) {
                    int exp = 1;
                    if (exp > maxExp[temp])
                        maxExp[temp] = exp;
                    break;
                }
            }
            int exp = 0;
            while (temp % p == 0) {
                exp++;
                temp /= p;
            }
            if (exp > maxExp[p])
                maxExp[p] = exp;
        }
    }
    
    long long ans = 1;
    for (int i = 2; i <= MAX; i++) {
        if (maxExp[i] > 0) {
            ans = (ans * modExp(i, maxExp[i])) % MOD;
        }
    }
    
    free(primes);
    return ans;
}

int main() {
    int n;
    scanf("%d", &n);
    int arr[n];
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }
    
    printf("%lld", lcm(arr, n));
    return 0;
}