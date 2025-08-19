#include <stdlib.h>
#include <limits.h>

typedef long long ll;
typedef struct { ll key; int delta; } Event;

static int cmpEvent(const void *a, const void *b) {
    Event *A = (Event*)a;
    Event *B = (Event*)b;
    if (A->key < B->key) return -1;
    if (A->key > B->key) return 1;
    return 0;
}

int maxIntersectionCount(int* y, int ySize) {
    int n = ySize;
    if (n < 2) return 0;
    Event *events = (Event*)malloc(sizeof(Event) * 2 * (n - 1));
    int m = 0;
    for (int i = 1; i < n; ++i) {
        ll s = 2LL * (ll)y[i-1];
        ll e;
        if (i == n - 1) e = 2LL * (ll)y[i];
        else e = 2LL * (ll)y[i] + (y[i] > y[i-1] ? -1LL : 1LL);
        ll lo = s < e ? s : e;
        ll hi = s < e ? e : s;
        events[m].key = lo; events[m].delta = 1; ++m;
        events[m].key = hi + 1; events[m].delta = -1; ++m;
    }
    qsort(events, m, sizeof(Event), cmpEvent);
    int cur = 0, ans = 0;
    int i = 0;
    while (i < m) {
        ll k = events[i].key;
        int sum = 0;
        while (i < m && events[i].key == k) {
            sum += events[i].delta;
            ++i;
        }
        cur += sum;
        if (cur > ans) ans = cur;
    }
    free(events);
    return ans;
}
