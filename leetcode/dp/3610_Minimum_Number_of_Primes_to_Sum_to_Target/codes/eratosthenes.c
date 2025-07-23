#include <stdio.h>
#include <limits.h>
#include <stdbool.h>

int minNumberOfPrimes(int n, int m) {
    int primes[1000], count = 0;
    int size = 8000;
    bool isPrime[8000];
    for (int i = 0; i < size; ++i) isPrime[i] = true;
    isPrime[0] = isPrime[1] = false;

    for (int i = 2; i * i < size; ++i) {
        if (isPrime[i]) {
            for (int j = i * i; j < size; j += i)
                isPrime[j] = false;
        }
    }

    for (int i = 2; i < size && count < m; ++i) {
        if (isPrime[i])
            primes[count++] = i;
    }

    int dp[1001];
    for (int i = 0; i <= n; ++i)
        dp[i] = INT_MAX;
    dp[0] = 0;

    for (int i = 0; i < m; ++i) {
        for (int j = primes[i]; j <= n; ++j) {
            if (dp[j - primes[i]] != INT_MAX && dp[j - primes[i]] + 1 < dp[j])
                dp[j] = dp[j - primes[i]] + 1;
        }
    }

    return dp[n] == INT_MAX ? -1 : dp[n];
}
