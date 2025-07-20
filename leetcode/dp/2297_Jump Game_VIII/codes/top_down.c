#include <stdio.h>
#include <limits.h>

#define MAXN 100005

int nse[MAXN], nge[MAXN];
long long memo[MAXN];

int globalNums[MAXN];
int globalCosts[MAXN];
int n;

long long dp(int i) {
    if (i == n) return LLONG_MAX;
    if (memo[i] != -1) return memo[i];
    if (i == n - 1) return memo[i] = (i == 0 ? 0 : globalCosts[i]);
    long long cost = (i == 0 ? 0 : globalCosts[i]);
    long long left = dp(nse[i]);
    long long right = dp(nge[i]);
    long long min_jump = (left < right ? left : right);
    return memo[i] = cost + min_jump;
}

long long minCost(int* nums, int numsSize, int* costs, int costsSize) {
    n = numsSize;
    for (int i = 0; i < n; ++i) {
        globalNums[i] = nums[i];
        globalCosts[i] = costs[i];
        nse[i] = nge[i] = n;
        memo[i] = -1;
    }

    int max_st[MAXN], min_st[MAXN];
    int top_max = -1, top_min = -1;

    for (int i = 0; i < n; ++i) {
        while (top_max >= 0 && globalNums[max_st[top_max]] <= globalNums[i]) {
            nge[max_st[top_max--]] = i;
        }
        while (top_min >= 0 && globalNums[min_st[top_min]] > globalNums[i]) {
            nse[min_st[top_min--]] = i;
        }
        max_st[++top_max] = i;
        min_st[++top_min] = i;
    }

    return dp(0);
}
