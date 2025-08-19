#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int idx;
    int k;
} QueryIndex;

int cmp_price(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int cmp_query(const void* a, const void* b) {
    return ((QueryIndex*)a)->k - ((QueryIndex*)b)->k;
}

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
long long* minimumRelativeLosses(int* prices, int pricesSize, int** queries, int queriesSize, int* queriesColSize, int* returnSize) {
    qsort(prices, pricesSize, sizeof(int), cmp_price);

    long long* prefix = (long long*)malloc((pricesSize + 1) * sizeof(long long));
    prefix[0] = 0;
    for (int i = 1; i <= pricesSize; i++)
        prefix[i] = prefix[i - 1] + prices[i - 1];

    QueryIndex* sortedQueries = (QueryIndex*)malloc(queriesSize * sizeof(QueryIndex));
    for (int i = 0; i < queriesSize; i++) {
        sortedQueries[i].idx = i;
        sortedQueries[i].k = queries[i][0];
    }
    qsort(sortedQueries, queriesSize, sizeof(QueryIndex), cmp_query);

    long long* result = (long long*)malloc(queriesSize * sizeof(long long));
    int j = 0;

    for (int q = 0; q < queriesSize; q++) {
        int i = sortedQueries[q].idx;
        int p = queries[i][0], num = queries[i][1];
        int pp = p << 1;

        while (j < pricesSize && prices[j] <= p) j++;
        int num1 = num < j ? num : j;
        int maxMid = (num1 < pricesSize - num) ? num1 : pricesSize - num;

        int left = 1, right = maxMid;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (prices[num1 - mid] > pp - prices[pricesSize - (num - num1) - mid])
                left = mid + 1;
            else
                right = mid - 1;
        }

        num1 -= (left - 1);
        int num2 = num - num1;

        long long bob = (long long)num2 * pp + prefix[num1];
        long long alice = prefix[pricesSize] - prefix[pricesSize - num2];
        result[i] = bob - alice;
    }

    free(sortedQueries);
    free(prefix);
    *returnSize = queriesSize;
    return result;
}
