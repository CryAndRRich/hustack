#include <stdio.h>

long long C2(long long x) {
    return x >= 2 ? x * (x - 1) / 2 : 0;
}

long long distributeCandies(int n, int limit) {
    long long total = C2(n + 2);
    long long sub1 = 3 * C2(n - (limit + 1) + 2);
    long long add2 = 3 * C2(n - 2 * (limit + 1) + 2);
    long long sub3 = C2(n - 3 * (limit + 1) + 2);
    return total - sub1 + add2 - sub3;
}