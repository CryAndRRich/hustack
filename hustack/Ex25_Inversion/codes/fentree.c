#include <stdio.h>
#include <string.h>

#define MAXV 1000005
#define MOD 1000000007

int bit[MAXV];

int query(int x) {
    int sum = 0;
    while (x > 0) {
        sum = (sum + bit[x]) % MOD;
        x -= x & -x;
    }
    return sum;
}

void update(int x) {
    while (x < MAXV) {
        bit[x] = (bit[x] + 1) % MOD;
        x += x & -x;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int a[n];
    for (int i = 0; i < n; i++) scanf("%d", &a[i]);

    long long res = 0;
    for (int i = n - 1; i >= 0; i--) {
        res = (res + query(a[i] - 1)) % MOD;
        update(a[i]);
    }

    printf("%lld", res);
    return 0;
}
