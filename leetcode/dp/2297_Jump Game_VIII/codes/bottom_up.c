#include <limits.h>
#include <stdlib.h>

long long minCost(int* nums, int numsSize, int* costs, int costsSize) {
    int n = numsSize;
    int* nse = malloc(n * sizeof(int));
    int* nge = malloc(n * sizeof(int));
    long long* dp = malloc(n * sizeof(long long));
    int* max_st = malloc(n * sizeof(int));
    int* min_st = malloc(n * sizeof(int));
    int top1 = -1, top2 = -1;

    for (int i = 0; i < n; i++) {
        nse[i] = n;
        nge[i] = n;
        dp[i] = LLONG_MAX;
    }

    for (int i = 0; i < n; i++) {
        while (top1 >= 0 && nums[max_st[top1]] <= nums[i]) {
            nge[max_st[top1--]] = i;
        }
        while (top2 >= 0 && nums[min_st[top2]] > nums[i]) {
            nse[min_st[top2--]] = i;
        }
        max_st[++top1] = i;
        min_st[++top2] = i;
    }

    dp[0] = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 2; j++) {
            int next = (j == 0) ? nse[i] : nge[i];
            if (next < n) {
                if (dp[next] > dp[i] + costs[next]) {
                    dp[next] = dp[i] + costs[next];
                }
            }
        }
    }

    long long result = dp[n - 1];
    free(nse); 
    free(nge); 
    free(dp); 
    free(max_st); 
    free(min_st);
    return result;
}
