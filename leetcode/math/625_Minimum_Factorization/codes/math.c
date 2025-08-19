#include <limits.h>

int cmp(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int smallestFactorization(int num) {
    if (num < 10) return num;

    int factors[32], count = 0;

    for (int d = 9; d >= 2; d--) {
        while (num % d == 0) {
            factors[count++] = d;
            num /= d;
        }
    }

    if (num != 1) return 0;

    qsort(factors, count, sizeof(int), cmp);

    long long result = 0;
    for (int i = 0; i < count; i++) {
        result = result * 10 + factors[i];
        if (result > INT_MAX) return 0;
    }

    return (int)result;
}
