#include <stdio.h>
#include <stdlib.h>

#define MAXN 100005

int bit[MAXN];

typedef struct {
    int val;
    int idx;
} Pair;

int cmp(const void* a, const void* b) {
    int x = *(int*)a;
    int y = *(int*)b;
    return (x > y) - (x < y);
}

int cmp_pair(const void* a, const void* b) {
    return ((Pair*)a)->val - ((Pair*)b)->val;
}

void update(int i, int n, int delta) {
    i++;
    while (i <= n + 1) {
        bit[i] += delta;
        i += i & -i;
    }
}

int query(int i) {
    i++;
    int res = 0;
    while (i) {
        res += bit[i];
        i -= i & -i;
    }
    return res;
}

int range_query(int l, int r) {
    return query(r) - query(l - 1);
}

int find_index(Pair* map, int n, int val) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (map[mid].val == val) return map[mid].idx;
        else if (map[mid].val < val) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

long long countOperationsToEmptyArray(int* nums, int numsSize) {
    for (int i = 0; i <= numsSize + 1; i++) bit[i] = 0;

    for (int i = 0; i < numsSize; i++) update(i, numsSize, 1);

    int* sorted = (int*)malloc(numsSize * sizeof(int));
    Pair* index_map = (Pair*)malloc(numsSize * sizeof(Pair));

    for (int i = 0; i < numsSize; i++) {
        sorted[i] = nums[i];
        index_map[i].val = nums[i];
        index_map[i].idx = i;
    }

    qsort(sorted, numsSize, sizeof(int), cmp);
    qsort(index_map, numsSize, sizeof(Pair), cmp_pair);

    long long total = 0;
    int prev_pos = 0;

    for (int i = 0; i < numsSize; i++) {
        int num = sorted[i];
        int pos = find_index(index_map, numsSize, num);
        int ops;
        if (pos >= prev_pos)
            ops = range_query(prev_pos, pos);
        else
            ops = range_query(prev_pos, numsSize - 1) + range_query(0, pos);
        total += ops;
        update(pos, numsSize, -1);
        prev_pos = pos;
    }

    free(sorted);
    free(index_map);
    return total;
}
