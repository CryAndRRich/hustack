#include <stdio.h>
#define MOD 1000000007
typedef unsigned long long ll;

ll inverse(ll a, ll mod) {
    ll res = 1, b = mod - 2;
    while (b > 0) {
        if (b % 2 == 1) {
            res = (res * a) % mod;
        }
        a = (a * a) % mod;
        b /= 2;
    }
    return res;
}

ll C(int k, int n) {
    if (k == 0 || k == n) {
        return 1;
    }
    if (k * 2 > n) {
        k = n - k;
    }

    ll res = 1;
    for (int i = 1; i <= k; i++) {
        res = (res * (n - k + i)) % MOD;
        res = (res * inverse(i, MOD)) % MOD;
    }
    
    return res;
}

int main() {
    int k, n;
    scanf("%d %d", &k, &n);
    printf("%lld\n", C(k, n));
    return 0;
}
