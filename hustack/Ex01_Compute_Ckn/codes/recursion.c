#include<stdio.h>
#include<math.h>
#define MOD 1000000007
typedef unsigned long long ll;
ll C[1000][1000];

void init () {
    for (int i = 0; i < 1000; i++)
        for (int j = 0; j < 1000; j++)
            C[i][j] = -1;
}
ll com(int k, int n) {
    if (k == 0) return 1;
    if (n == 0) return 0; 
    if (C[k][n] == -1) 
        C[k][n] = (com(k, n - 1) + com(k - 1, n - 1)) % MOD; 
    return C[k][n];
}
int main () {
    init();
    int n, k;
    scanf("%d %d", &k, &n);
    printf("%llu", com(k, n));
}
