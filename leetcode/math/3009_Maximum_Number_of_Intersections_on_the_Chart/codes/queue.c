#include <stdio.h>
#include <stdlib.h>

typedef struct {
    long long l, r;
} Pair;

int cmpPair(const void* a, const void* b) {
    long long diff = ((Pair*)a) -> l - ((Pair*)b) -> l;
    return (diff < 0) ? -1 : (diff > 0);
}

typedef struct {
    long long *arr;
    int size;
    int cap;
} MinHeap;

MinHeap* createHeap(int cap) {
    MinHeap* h = (MinHeap*)malloc(sizeof(MinHeap));
    h -> arr = (long long*)malloc(sizeof(long long) * cap);
    h -> size = 0;
    h -> cap = cap;
    return h;
}

void swapLL(long long* a, long long* b) {
    long long t = *a; *a = *b; *b = t;
}

void heapPush(MinHeap* h, long long val) {
    int i = h -> size++;
    h -> arr[i] = val;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h -> arr[p] <= h -> arr[i]) break;
        swapLL(&h -> arr[p], &h -> arr[i]);
        i = p;
    }
}

long long heapTop(MinHeap* h) {
    return h -> arr[0];
}

void heapPop(MinHeap* h) {
    if (h -> size == 0) return;
    h -> arr[0] = h -> arr[--h -> size];
    int i = 0;
    while (1) {
        int l = i * 2 + 1, r = i * 2 + 2, smallest = i;
        if (l < h -> size && h -> arr[l] < h -> arr[smallest]) smallest = l;
        if (r < h -> size && h -> arr[r] < h -> arr[smallest]) smallest = r;
        if (smallest == i) break;
        swapLL(&h -> arr[i], &h -> arr[smallest]);
        i = smallest;
    }
}

int maxIntersectionCount(int* y, int ySize) {
    int res = 0;
    Pair* candles = (Pair*)malloc(sizeof(Pair) * (ySize - 1));
    int cnt = 0;
    for (int i = 0; i + 1 < ySize; ++i) {
        long long l = 2LL * y[i] + (i ? (y[i] < y[i + 1] ? 1 : -1) : 0);
        long long r = 2LL * y[i + 1];
        candles[cnt].l = (l < r) ? l : r;
        candles[cnt].r = (l < r) ? r : l;
        cnt++;
    }
    qsort(candles, cnt, sizeof(Pair), cmpPair);
    MinHeap* pq = createHeap(cnt);
    for (int i = 0; i < cnt; ++i) {
        heapPush(pq, candles[i].r);
        while (pq -> size && heapTop(pq) < candles[i].l)
            heapPop(pq);
        if (pq -> size > res) res = pq -> size;
    }
    free(candles);
    free(pq -> arr);
    free(pq);
    return res;
}
