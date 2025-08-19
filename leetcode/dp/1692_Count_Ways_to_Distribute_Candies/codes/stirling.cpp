#include <bits/stdc++.h>
using namespace std;
const int MOD = 1000000007;

long long modpow(long long a, long long e) {
    long long r = 1;
    while (e) {
        if (e & 1) r = r * a % MOD;
        a = a * a % MOD;
        e >>= 1;
    }
    return r;
}

pair<vector<long long>, vector<long long>> build_fact_and_ifact(int n) {
    vector<long long> fact(n + 1, 1), ifact(n + 1, 1);
    for (int i = 1; i <= n; i++) fact[i] = fact[i - 1] * i % MOD;
    ifact[n] = modpow(fact[n], MOD - 2);
    for (int i = n; i > 0; i--) ifact[i - 1] = ifact[i] * i % MOD;
    return {fact, ifact};
}

long long nCk_mod(int n, int k, vector<long long> &fact, vector<long long> &ifact) {
    if (k < 0 || k > n) return 0;
    return fact[n] * ifact[k] % MOD * ifact[n - k] % MOD;
}

long long stirling2_mod(int n, int k) {
    if (k < 0 || k > n) return 0;
    auto fi = build_fact_and_ifact(k);
    auto fact = fi.first;
    auto ifact = fi.second;
    long long fact_k = fact[k];
    long long total = 0;
    for (int m = 0; m <= k; m++) {
        long long binom = nCk_mod(k, m, fact, ifact);
        long long term = binom * modpow(k - m, n) % MOD;
        if (m & 1) total = (total - term + MOD) % MOD;
        else total = (total + term) % MOD;
    }
    long long inv_fact_k = modpow(fact_k, MOD - 2);
    return total * inv_fact_k % MOD;
}

class Solution {
public:
    int waysToDistribute(int n, int k) {
        return stirling2_mod(n, k);
    }
};
