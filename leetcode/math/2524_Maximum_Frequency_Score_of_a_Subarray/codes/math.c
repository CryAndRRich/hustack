#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOD 1000000007
#define MAX_VAL 1000000

typedef long long i64;

int modPlus(int a, int b) {
    return ((i64)a + b) % MOD;
}

int modMinus(int a, int b) {
    return ((i64)a - b + MOD) % MOD;
}

int modMult(int a, int b) {
    return ((i64)a * b) % MOD;
}

int modPow(int base, int exp) {
    i64 result = 1;
    while (exp > 0) {
        if (exp & 1) result = modMult(result, base);
        exp >>= 1;
        base = modMult(base, base);
    }
    return result;
}

int maxFrequencyScore(int* nums, int n, int k) {
    int* freq = (int*)calloc(MAX_VAL + 1, sizeof(int));
    int result = 0, sum = 0;

    for (int r = 0; r < n; ++r) {
        int val = nums[r];
        int prevExp = freq[val]++;
        if (prevExp > 0) {
            int contribution = modPow(val, prevExp);
            sum = modMinus(sum, contribution);
            contribution = modMult(contribution, val);
            sum = modPlus(sum, contribution);
        } else {
            sum = modPlus(sum, val);
        }

        if (r >= k) {
            int leftVal = nums[r - k];
            int nextExp = --freq[leftVal];
            if (nextExp > 0) {
                int contribution = modPow(leftVal, nextExp + 1);
                sum = modMinus(sum, contribution);
                contribution = modPow(leftVal, nextExp);
                sum = modPlus(sum, contribution);
            } else {
                sum = modMinus(sum, leftVal);
            }
        }

        if (r >= k - 1) {
            if (sum > result) result = sum;
        }
    }

    free(freq);
    return result;
}
