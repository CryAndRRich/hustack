#include <stdio.h>
#include <stdlib.h>

int cmp(const void* a, const void* b) {
    return ((int*)a)[0] - ((int*)b)[0];
}

int upper_bound(int arr[][2], int n, int x) {
    int l = 0, r = n;
    while (l < r) {
        int m = (l + r) / 2;
        if (arr[m][0] <= x) l = m + 1;
        else r = m;
    }
    return l;
}

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
long long* maximumCoins(int* heroes, int heroesSize, int* monsters, int monstersSize, int* coins, int coinsSize, int* returnSize) {
    int (*mon_coin)[2] = malloc(sizeof(int) * 2 * monstersSize);
    for (int i = 0; i < monstersSize; ++i) {
        mon_coin[i][0] = monsters[i];
        mon_coin[i][1] = coins[i];
    }
    qsort(mon_coin, monstersSize, sizeof(mon_coin[0]), cmp);

    long long* prefix = malloc(sizeof(long long) * (monstersSize + 1));
    prefix[0] = 0;
    for (int i = 0; i < monstersSize; ++i) {
        prefix[i + 1] = prefix[i] + mon_coin[i][1];
    }

    long long* ans = malloc(sizeof(long long) * heroesSize);
    for (int i = 0; i < heroesSize; ++i) {
        int idx = upper_bound(mon_coin, monstersSize, heroes[i]);
        ans[i] = prefix[idx];
    }

    free(mon_coin);
    free(prefix);
    *returnSize = heroesSize;
    return ans;
}
