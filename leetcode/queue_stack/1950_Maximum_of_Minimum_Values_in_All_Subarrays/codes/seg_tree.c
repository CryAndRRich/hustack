#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MAXN 2005

int seg[4 * MAXN];
int nums[MAXN];

void build(int node, int l, int r) {
    if (l == r) {
        seg[node] = nums[l];
    } else {
        int mid = (l + r) / 2;
        build(2 * node + 1, l, mid);
        build(2 * node + 2, mid + 1, r);
        seg[node] = seg[2 * node + 1] < seg[2 * node + 2] ? seg[2 * node + 1] : seg[2 * node + 2];
    }
}

int query(int node, int l, int r, int ql, int qr) {
    if (qr < l || ql > r) return INT_MAX;
    if (ql <= l && r <= qr) return seg[node];
    int mid = (l + r) / 2;
    int left = query(2 * node + 1, l, mid, ql, qr);
    int right = query(2 * node + 2, mid + 1, r, ql, qr);
    return left < right ? left : right;
}

int* findMaximums(int* input, int n, int* returnSize) {
    for (int i = 0; i < n; ++i) nums[i] = input[i];
    build(0, 0, n - 1);

    int* res = (int*)malloc(sizeof(int) * n);
    for (int k = 1; k <= n; ++k) {
        int max_min = 0;
        for (int i = 0; i + k <= n; ++i) {
            int min_val = query(0, 0, n - 1, i, i + k - 1);
            if (min_val > max_min) max_min = min_val;
        }
        res[k - 1] = max_min;
    }

    *returnSize = n;
    return res;
}
