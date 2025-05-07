#include <stdio.h>
#include <stdlib.h>

int cmp_int(const void* a, const void* b){
    int x = *(const int*)a, y = *(const int*)b;
    return x - y;
}

int lower_bound(int *arr, int n, int x){
    int l = 0, r = n;
    while (l < r){
        int m = (l + r) >> 1;
        if (arr[m] < x) l = m + 1;
        else r = m;
    }
    return l;
}

int *tree;
int N;

void seg_update(int pos, int val){
    pos += N;
    if (tree[pos] < val) tree[pos] = val;
    for (pos >>= 1; pos; pos >>= 1){
        if (tree[pos] < tree[2*pos]) tree[pos] = tree[2*pos];
        if (tree[pos] < tree[2*pos+1]) tree[pos] = tree[2*pos+1];
    }
}

int seg_query(int l, int r){
    int res = 0;
    l += N; r += N;
    while (l <= r){
        if (l & 1) { if (res < tree[l]) res = tree[l]; l++; }
        if (!(r & 1)){ if (res < tree[r]) res = tree[r]; r--; }
        l >>= 1; r >>= 1;
    }
    return res;
}

int main(){
    int n;
    scanf("%d", &n);
    int *a = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &a[i]);

    int *b = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) b[i] = a[i];
    qsort(b, n, sizeof(int), cmp_int);
    int m = 1;
    for (int i = 1; i < n; i++)
        if (b[i] != b[m-1]) b[m++] = b[i];

    N = 1; 
    while (N < m) N <<= 1;
    tree = calloc(2*N, sizeof(int));

    int best = 0;
    for (int i = 0; i < n; i++){
        int idx = lower_bound(b, m, a[i]);
        int cur = 1;
        if (idx > 0) cur = seg_query(0, idx-1) + 1;
        seg_update(idx, cur);
        if (cur > best) best = cur;
    }

    printf("%d", best);
    return 0;
}
