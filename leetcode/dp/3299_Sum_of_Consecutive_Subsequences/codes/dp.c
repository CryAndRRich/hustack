#include <stdio.h>
#include <stdlib.h>

#define MOD 1000000007LL

int getSum(int* nums, int numsSize) {
    int mx = 0;
    for (int i = 0; i < numsSize; ++i) if (nums[i] > mx) mx = nums[i];

    long long *dp_inc_before = calloc(mx + 5, sizeof(long long));
    long long *dp_dec_before = calloc(mx + 5, sizeof(long long));
    long long *left_inc = calloc(numsSize, sizeof(long long));
    long long *left_dec = calloc(numsSize, sizeof(long long));

    for (int i = 0; i < numsSize; ++i) {
        int x = nums[i];
        if (x - 1 >= 0) left_inc[i] = dp_inc_before[x - 1];
        if (x + 1 < mx + 5) left_dec[i] = dp_dec_before[x + 1];
        long long inc_end = (1 + left_inc[i]) % MOD;
        long long dec_end = (1 + left_dec[i]) % MOD;
        dp_inc_before[x] = (dp_inc_before[x] + inc_end) % MOD;
        d
        p_dec_before[x] = (dp_dec_before[x] + dec_end) % MOD;
    }
    long long *dp_inc_after = calloc(mx + 5, sizeof(long long));
    long long *dp_dec_after = calloc(mx + 5, sizeof(long long));
    long long *right_inc = calloc(numsSize, sizeof(long long));
    long long *right_dec = calloc(numsSize, sizeof(long long));
    for (int i = numsSize - 1; i >= 0; --i) {
        int x = nums[i];
        if (x + 1 < mx + 5) right_inc[i] = dp_inc_after[x + 1];
        if (x - 1 >= 0) right_dec[i] = dp_dec_after[x - 1];
        long long inc_start = (1 + right_inc[i]) % MOD;
        long long dec_start = (1 + right_dec[i]) % MOD;
        dp_inc_after[x] = (dp_inc_after[x] + inc_start) % MOD;
        dp_dec_after[x] = (dp_dec_after[x] + dec_start) % MOD;
    }

    long long ans = 0;
    for (int i = 0; i < numsSize; ++i) {
        long long inc_incl = ((1 + left_inc[i]) % MOD) * ((1 + right_inc[i]) % MOD) % MOD;
        long long dec_incl = ((1 + left_dec[i]) % MOD) * ((1 + right_dec[i]) % MOD) % MOD;
        long long total = (inc_incl + dec_incl - 1) % MOD;
        if (total < 0) total += MOD;
        ans = (ans + (long long)nums[i] * total) % MOD;
    }

    free(dp_inc_before);
    free(dp_dec_before);
    free(left_inc);
    free(left_dec);
    free(dp_inc_after);
    free(dp_dec_after);
    free(right_inc);
    free(right_dec);
    
    return (int)ans;
}
