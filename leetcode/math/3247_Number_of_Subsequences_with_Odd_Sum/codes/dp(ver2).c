#include <stdio.h>

#define MOD 1000000007

int subsequenceCount(int* nums, int numsSize) {
    long long even = 0, odd = 0;

    for (int i = 0; i < numsSize; i++) {
        if (nums[i] % 2 == 0) {
            even = (even * 2 + 1) % MOD;
            odd = (odd * 2) % MOD;
        } else {
            long long new_even = (even + odd) % MOD;
            long long new_odd = (even + odd + 1) % MOD;
            even = new_even;
            odd = new_odd;
        }
    }

    return (int) odd;
}
