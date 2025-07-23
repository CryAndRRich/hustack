#include <bits/stdc++.h>
using namespace std;

const int MOD = 998244353;
const int PRIMITIVE_ROOT = 3;

class Solution {
public:
    int mod_pow(int a, int b, int mod) {
        int res = 1;
        while (b > 0) {
            if (b & 1)
                res = 1LL * res * a % mod;
            a = 1LL * a * a % mod;
            b >>= 1;
        }
        return res;
    }

    void ntt(vector<int>& a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; ++i) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
                swap(a[i], a[j]);
        }

        for (int len = 2; len <= n; len <<= 1) {
            int wlen = mod_pow(PRIMITIVE_ROOT, (MOD - 1) / len, MOD);
            if (invert)
                wlen = mod_pow(wlen, MOD - 2, MOD);
            for (int i = 0; i < n; i += len) {
                int w = 1;
                for (int j = 0; j < len / 2; ++j) {
                    int u = a[i + j], v = 1LL * a[i + j + len / 2] * w % MOD;
                    a[i + j] = (u + v) % MOD;
                    a[i + j + len / 2] = (u - v + MOD) % MOD;
                    w = 1LL * w * wlen % MOD;
                }
            }
        }

        if (invert) {
            int inv_n = mod_pow(n, MOD - 2, MOD);
            for (int& x : a)
                x = 1LL * x * inv_n % MOD;
        }
    }

    vector<long long> multiply(vector<int>& a, vector<int>& b) {
        int n = 1;
        while (n < a.size() + b.size() - 1) n <<= 1;
        a.resize(n);
        b.resize(n);
        ntt(a, false);
        ntt(b, false);
        for (int i = 0; i < n; ++i)
            a[i] = 1LL * a[i] * b[i] % MOD;
        ntt(a, true);
        a.resize(a.size() + b.size() - 1 - (n - n));
        return a;
    }
};
