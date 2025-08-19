#include <stdio.h>
#include <stdlib.h>

#define MOD 1000000007

int subsequenceCount(int* nums, int numsSize) {
    long long** dp = (long long**)malloc(numsSize * sizeof(long long*));
    for (int i = 0; i < numsSize; i++) {
        dp[i] = (long long*)calloc(2, sizeof(long long));
    }

    if (nums[0] % 2 == 0) {
        dp[0][0] = 1;  
        dp[0][1] = 0;
    } else {
        dp[0][0] = 0;
        dp[0][1] = 1;
    }

    for (int i = 1; i < numsSize; i++) {
        if (nums[i] % 2 == 0) {
            dp[i][0] = (dp[i - 1][0] * 2 + 1) % MOD;
            dp[i][1] = (dp[i - 1][1] * 2) % MOD;
        } else {
            dp[i][0] = (dp[i - 1][0] + dp[i - 1][1]) % MOD;
            dp[i][1] = (dp[i - 1][0] + dp[i - 1][1] + 1) % MOD;
        }
    }

    int result = dp[numsSize - 1][1];

    for (int i = 0; i < numsSize; i++) {
        free(dp[i]);
    }
    free(dp);

    return result;
}
