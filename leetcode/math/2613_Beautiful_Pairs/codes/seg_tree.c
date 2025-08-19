#include <stdlib.h>
#include <limits.h>

typedef struct {
    int val;
    int idx;
} Pair;

int cmpPair(const void *a, const void *b) {
    const Pair *pa = (const Pair*)a;
    const Pair *pb = (const Pair*)b;
    if (pa->val != pb->val) return pa->val - pb->val;
    return pa->idx - pb->idx;
}

int better(const long *v, int x, int y) {
    if (x < 0) return y;
    if (y < 0) return x;
    if (v[x] < v[y] || (v[x] == v[y] && y < x)) return y;
    return x;
}

int query_tree(int *tree, const long *v, int ind, int l, int r, int ql, int qr) {
    if (ql > qr) return -1;
    if (l >= ql && r <= qr) return tree[ind];
    int mid = (l + r) >> 1;
    int res = -1;
    if (ql <= mid) res = query_tree(tree, v, ind<<1, l, mid, ql, qr);
    if (qr > mid) res = better(v, res, query_tree(tree, v, (ind<<1)|1, mid+1, r, ql, qr));
    return res;
}

void update_tree(int *tree, const long *v, int ind, int l, int r, int pos, int val) {
    tree[ind] = better(v, tree[ind], val);
    if (l == r) return;
    int mid = (l + r) >> 1;
    if (pos <= mid) update_tree(tree, v, ind<<1, l, mid, pos, val);
    else update_tree(tree, v, (ind<<1)|1, mid+1, r, pos, val);
}

int* beautifulPair(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize) {
    int n = nums1Size;
    long *p1 = malloc(sizeof(long)*n);
    long *p2 = malloc(sizeof(long)*n);
    int i;
    int maxCoord = 0;
    for (i = 0; i < n; ++i) {
        p1[i] = (long)nums2[i] + (long)nums1[i];
        p2[i] = (long)nums2[i] - (long)nums1[i];
        if (nums1[i] > maxCoord) maxCoord = nums1[i];
    }
    int size = maxCoord + 2;
    int treeSize = 4*(size+5);
    int *tree1 = malloc(sizeof(int)*treeSize);
    int *tree2 = malloc(sizeof(int)*treeSize);
    for (i = 0; i < treeSize; ++i) { tree1[i] = -1; tree2[i] = -1; }

    Pair *arr = malloc(sizeof(Pair)*n);
    for (i = 0; i < n; ++i) {
        arr[i].val = nums2[i];
        arr[i].idx = i;
    }
    qsort(arr, n, sizeof(Pair), cmpPair);

    int ans0 = n, ans1 = n;
    long long d = LLONG_MAX;
    for (i = 0; i < n; ++i) {
        int id = arr[i].idx;
        int x = nums1[id];
        if (x >= 0) {
            int i1 = query_tree(tree1, p1, 1, 0, maxCoord+1, 0, x);
            if (i1 >= 0) {
                long long may = p1[id] - p1[i1];
                if (may < d || (may == d && ( ( (id<i1?id:i1) < ans0) || (((id<i1?id:i1) == ans0) && ((id>i1?id:i1) < ans1)) ))) {
                    d = may; ans0 = id<i1?id:i1; ans1 = id>i1?id:i1;
                }
            }
        }
        int i2 = query_tree(tree2, p2, 1, 0, maxCoord+1, x, maxCoord+1);
        if (i2 >= 0) {
            long long may = p2[id] - p2[i2];
            if (may < d || (may == d && ( ( (id<i2?id:i2) < ans0) || (((id<i2?id:i2) == ans0) && ((id>i2?id:i2) < ans1)) ))) {
                d = may; ans0 = id<i2?id:i2; ans1 = id>i2?id:i2;
            }
        }
        update_tree(tree1, p1, 1, 0, maxCoord+1, x, id);
        update_tree(tree2, p2, 1, 0, maxCoord+1, x, id);
    }
    free(p1); free(p2); free(tree1); free(tree2); free(arr);

    int *res = malloc(sizeof(int)*2);
    res[0] = ans0; res[1] = ans1;
    *returnSize = 2;
    return res;
}
