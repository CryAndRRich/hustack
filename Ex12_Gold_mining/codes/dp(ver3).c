#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int size;
    int *tree;
} SegmentTree;

SegmentTree* createSegmentTree(int n) {
    SegmentTree *seg = (SegmentTree *)malloc(sizeof(SegmentTree));
    seg -> size = 1;
    while (seg -> size < n)
        seg -> size *= 2;
    seg -> tree = (int*)malloc(2 * seg -> size * sizeof(int));
    for (int i = 0; i < 2 * seg -> size; i++)
        seg -> tree[i] = 0;
    return seg;
}

void update(SegmentTree *seg, int ind, int val) {
    ind += seg -> size;
    seg -> tree[ind] = val;
    while (ind > 1) {
        ind /= 2;
        int l = seg -> tree[2 * ind];
        int r = seg -> tree[2 * ind + 1];
        seg -> tree[ind] = (l > r ? l : r);
    }
}

int query(SegmentTree *seg, int l, int r) {
    int res = 0;
    l += seg -> size;
    r += seg -> size;
    while (l <= r) {
        if (l % 2 == 1) {
            if (seg -> tree[l] > res)
                res = seg -> tree[l];
            l++;
        }
        if (r % 2 == 0) {
            if (seg -> tree[r] > res)
                res = seg -> tree[r];
            r--;
        }
        l /= 2;
        r /= 2;
    }
    return res;
}

int main(void) {
    int n, L1, L2;
    scanf("%d %d %d", &n, &L1, &L2);
    
    int *a = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        scanf("%d", &a[i]);
    }
    
    int *dp = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        dp[i] = 0;
    int ans = 0;
    
    SegmentTree *seg = createSegmentTree(n);
    
    for (int i = 0; i < n; i++) {
        int best_prev = 0;
        if (i - L1 >= 0) {
            int l = (i - L2 < 0) ? 0 : i - L2;
            int r = i - L1;
            best_prev = query(seg, l, r);
        }
        dp[i] = a[i] + best_prev;
        if (dp[i] > ans)
            ans = dp[i];
        update(seg, i, dp[i]);
    }
    
    printf("%d", ans);
    
    free(a);
    free(dp);
    free(seg -> tree);
    free(seg);
    return 0;
}
