#include <iostream>
using namespace std;

const unsigned long long MOD = 1000000007;
unsigned long long C[1000][1000];

void init() {
    for (int i = 0; i < 1000; i++)
        for (int j = 0; j < 1000; j++)
            C[i][j] = -1;
}

unsigned long long com(int k, int n) {
    if (k == 0)
        return 1;
    if (n == 0)
        return 0;
    if (C[k][n] == -1)
        C[k][n] = (com(k, n - 1) + com(k - 1, n - 1)) % MOD;
    return C[k][n];
}

int main() {
    init();
    int n, k;
    cin >> k >> n;
    cout << com(k, n);
    return 0;
}
