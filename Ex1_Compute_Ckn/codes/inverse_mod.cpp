#include <iostream>
using namespace std;

const long long MOD = 1000000007;

long long inverse(long long a, long long mod) {
    long long res = 1, b = mod - 2;
    while (b > 0) {
        if (b % 2 == 1) {
            res = (res * a) % mod;
        }
        a = (a * a) % mod;
        b /= 2;
    }
    return res;
}

long long C(int k, int n) {
    if (k == 0 || k == n) {
        return 1;
    }
    if (k * 2 > n) {
        k = n - k;
    }
    
    long long res = 1;
    for (int i = 1; i <= k; i++) {
        res = (res * (n - k + i)) % MOD;
        res = (res * inverse(i, MOD)) % MOD;
    }
    
    return res;
}

int main() {
    int k, n;
    cin >> k >> n;
    cout << C(k, n) << "\n";
    return 0;
}
