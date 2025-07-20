#include <stdio.h>
#include <stdlib.h>

int cmp(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

int lower_bound(int* arr, int size, int val) {
    int l = 0, r = size;
    while (l < r) {
        int m = (l + r) / 2;
        if (arr[m] < val) l = m + 1;
        else r = m;
    }
    return l;
}

void update(int* tree, int size, int i, int delta) {
    while (i <= size) {
        tree[i] += delta;
        i += i & -i;
    }
}

int query(int* tree, int i) {
    int res = 0;
    while (i > 0) {
        res += tree[i];
        i -= i & -i;
    }
    return res;
}

int kBigIndices(int* nums, int n, int k) {
    int* sorted = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) sorted[i] = nums[i];
    qsort(sorted, n, sizeof(int), cmp);

    int* comp = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        comp[i] = lower_bound(sorted, n, nums[i]) + 1;

    int* left = calloc(n, sizeof(int));
    int* right = calloc(n, sizeof(int));
    int* bit = calloc(n + 2, sizeof(int));

    for (int i = 0; i < n; i++) {
        left[i] = query(bit, comp[i] - 1);
        update(bit, n + 1, comp[i], 1);
    }

    for (int i = 0; i <= n + 1; i++) bit[i] = 0;

    for (int i = n - 1; i >= 0; i--) {
        right[i] = query(bit, comp[i] - 1);
        update(bit, n + 1, comp[i], 1);
    }

    int res = 0;
    for (int i = 0; i < n; i++) {
        if (left[i] >= k && right[i] >= k)
            res++;
    }

    free(sorted); free(comp); free(left); free(right); free(bit);
    return res;
}
