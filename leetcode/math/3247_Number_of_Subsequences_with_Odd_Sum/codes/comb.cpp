#include <bits/stdc++.h>
using namespace std;

const int MOD = 1e9 + 7;
const int MAXN = 100005;

long long fact[MAXN], invfact[MAXN];

long long powmod(long long base, long long exp) {
    long long res = 1;
    while (exp) {
        if (exp % 2) res = res * base % MOD;
        base = base * base % MOD;
        exp /= 2;
    }
    return res;
}

void init_factorials() {
    fact[0] = 1;
    for (int i = 1; i < MAXN; i++)
        fact[i] = fact[i - 1] * i % MOD;
    invfact[MAXN - 1] = powmod(fact[MAXN - 1], MOD - 2);
    for (int i = MAXN - 2; i >= 0; i--)
        invfact[i] = invfact[i + 1] * (i + 1) % MOD;
}

long long comb(int n, int k) {
    if (k < 0 || k > n) return 0;
    return fact[n] * invfact[k] % MOD * invfact[n - k] % MOD;
}

class Solution {
public:
    int subsequenceCount(vector<int>& nums) {
        int even = 0, odd = 0;
        for (int x : nums) {
            if (x % 2 == 0) even++;
            else odd++;
        }

        init_factorials();

        long long res = 0;
        for (int k = 1; k <= odd; k += 2) {
            res = (res + comb(odd, k) * powmod(2, even)) % MOD;
        }

        return res;
    }
};
