#include <stdio.h>
#include <stdlib.h>

typedef long long ll;

typedef struct {
    int n;
    ll *bit;
} Fenwick;

Fenwick* fenwick_create(int n) {
    Fenwick *fw = (Fenwick*)malloc(sizeof(Fenwick));
    fw -> n = n;
    fw -> bit = (ll*)calloc(n + 1, sizeof(ll));
    return fw;
}
void fenwick_free(Fenwick *fw) {
    free(fw -> bit);
    free(fw);
}
void fenwick_update(Fenwick *fw, int idx, int delta) {
    int i = idx + 1;
    while (i <= fw -> n) {
        fw -> bit[i] += delta;
        i += i & -i;
    }
}
ll fenwick_query(Fenwick *fw, int idx) {
    if (idx < 0) return 0;
    ll s = 0;
    int i = idx + 1;
    while (i > 0) {
        s += fw -> bit[i];
        i -= i & -i;
    }
    return s;
}
ll fenwick_range(Fenwick *fw, int l, int r) {
    if (l > r) return 0;
    return fenwick_query(fw, r) - fenwick_query(fw, l - 1);
}

int cmp_int(const void *a, const void *b) {
    int x = *(int*)a, y = *(int*)b;
    if (x < y) return -1;
    if (x > y) return 1;
    return 0;
}

int lower_bound_arr(int *arr, int n, int x) {
    int l = 0, r = n;
    while (l < r) {
        int m = (l + r) >> 1;
        if (arr[m] < x) l = m + 1;
        else r = m;
    }
    return l;
}
int upper_bound_arr(int *arr, int n, long long x) {
    int l = 0, r = n;
    while (l < r) {
        int m = (l + r) >> 1;
        if ((long long)arr[m] <= x) l = m + 1;
        else r = m;
    }
    return l;
}

int check(int *nums, int n, int th, ll K, int *vals, int m) {
    Fenwick *fw = fenwick_create(m);
    ll cnt = 0;
    for (int i = 0; i < n; ++i) {
        int num = nums[i];
        int idxL = upper_bound_arr(vals, m, (long long)num);          
        int idxR = upper_bound_arr(vals, m, (long long)num + th) - 1; 
        if (idxL <= idxR) {
            cnt += fenwick_range(fw, idxL, idxR);
            if (cnt >= K) { fenwick_free(fw); return 1; }
        }
        int idxNum = lower_bound_arr(vals, m, num);
        fenwick_update(fw, idxNum, 1);
    }
    fenwick_free(fw);
    return cnt >= K;
}

int minThreshold(int *nums, int n, long long K) {
    if (n == 0) return -1;
    int *vals = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) vals[i] = nums[i];
    qsort(vals, n, sizeof(int), cmp_int);
    int m = 0;
    for (int i = 0; i < n; ++i) if (i == 0 || vals[i] != vals[i-1]) vals[m++] = vals[i];

    int lo = 0, hi = vals[m-1] - vals[0];
    int ans = -1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (check(nums, n, mid, K, vals, m)) {
            ans = mid;
            hi = mid - 1;
        } else lo = mid + 1;
    }
    free(vals);
    return ans;
}
