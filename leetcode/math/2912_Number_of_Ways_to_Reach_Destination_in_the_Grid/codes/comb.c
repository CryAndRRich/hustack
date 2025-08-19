#include <stdio.h>
#include <stdlib.h>

#define MOD 1000000007

int numberOfWays(int n, int m, int k, int* source, int sourceSize, int* dest, int destSize) {
    long long same = (source[0] == dest[0] && source[1] == dest[1]) ? 1 : 0;
    long long row  = (source[0] == dest[0] && source[1] != dest[1]) ? 1 : 0;
    long long col  = (source[1] == dest[1] && source[0] != dest[0]) ? 1 : 0;
    long long other= (source[0] != dest[0] && source[1] != dest[1]) ? 1 : 0;

    for (int step = 0; step < k; ++step) {
        long long new_same = (row + col) % MOD;
        long long new_row = (same * (m - 1) % MOD + row * (m - 2) % MOD + other * 1 % MOD) % MOD;
        long long new_col = (same * (n - 1) % MOD + col * (n - 2) % MOD + other * 1 % MOD) % MOD;
        long long new_other = (row * (n - 1) % MOD + col * (m - 1) % MOD + other * (n + m - 4) % MOD) % MOD;

        same = new_same;
        row = new_row;
        col = new_col;
        other = new_other;
    }
    return (int)same;
}
