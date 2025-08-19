#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int n;
    int* arr;
} SegmentTree;

SegmentTree createSegTree(int n) {
    SegmentTree st;
    st.n = n;
    st.arr = (int*)malloc(sizeof(int) * (2 * n));
    for (int i = 0; i < 2 * n; i++) st.arr[i] = -1;
    return st;
}

void updateSeg(SegmentTree* st, int idx, int val) {
    idx += st -> n;
    st -> arr[idx] = val;
    while (idx > 1) {
        idx >>= 1;
        int a = st -> arr[idx << 1], b = st -> arr[idx << 1 | 1];
        st -> arr[idx] = a > b ? a : b;
    }
}

int querySeg(SegmentTree* st, int l, int r) {
    if (l > r) return -1;
    l += st -> n;
    r += st -> n;
    int res = -1;
    while (l <= r) {
        if (l & 1) { if (st -> arr[l] > res) res = st -> arr[l]; l++; }
        if (!(r & 1)) { if (st -> arr[r] > res) res = st -> arr[r]; r--; }
        l >>= 1; r >>= 1;
    }
    return res;
}

int cmpInt(const void* a, const void* b) {
    int x = *(const int*)a, y = *(const int*)b;
    return (x > y) - (x < y);
}

int compress(int* arr, int n, int* out) {
    for (int i = 0; i < n; i++) out[i] = arr[i];
    qsort(out, n, sizeof(int), cmpInt);
    int m = 0;
    for (int i = 0; i < n; i++)
        if (i == 0 || out[i] != out[i - 1]) out[m++] = out[i];
    return m;
}

int getIndex(int* arr, int m, int val) {
    int l = 0, r = m - 1;
    while (l <= r) {
        int mid = (l + r) >> 1;
        if (arr[mid] == val) return mid;
        if (arr[mid] < val) l = mid + 1;
        else r = mid - 1;
    }
    return -1;
}

int maxProfit(int* prices, int pricesSize, int* profits, int profitsSize) {
    int* comp = (int*)malloc(sizeof(int) * pricesSize);
    int m = compress(prices, pricesSize, comp);

    int* maxleft = (int*)malloc(sizeof(int) * pricesSize);
    int* maxright = (int*)malloc(sizeof(int) * pricesSize);

    SegmentTree segleft = createSegTree(m);
    SegmentTree segright = createSegTree(m);

    for (int i = 0; i < pricesSize; i++) {
        int id = getIndex(comp, m, prices[i]);
        maxleft[i] = querySeg(&segleft, 0, id - 1);
        if (profits[i] > segleft.arr[id + m]) 
            updateSeg(&segleft, id, profits[i]);
    }

    for (int i = pricesSize - 1; i >= 0; i--) {
        int id = getIndex(comp, m, prices[i]);
        maxright[i] = querySeg(&segright, id + 1, m - 1);
        if (profits[i] > segright.arr[id + m])
            updateSeg(&segright, id, profits[i]);
    }

    int ans = -1;
    for (int i = 0; i < pricesSize; i++) {
        if (maxleft[i] >= 0 && maxright[i] >= 0) {
            int sum = profits[i] + maxleft[i] + maxright[i];
            if (sum > ans) ans = sum;
        }
    }

    free(comp);
    free(maxleft);
    free(maxright);
    free(segleft.arr);
    free(segright.arr);
    return ans;
}
