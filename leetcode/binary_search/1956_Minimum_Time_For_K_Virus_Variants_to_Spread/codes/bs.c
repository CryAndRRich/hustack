#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXP 1000
#define MAXE (MAXP * 4)
#define MAXX (MAXE + 5)
#define MAXY (MAXE + 5)

typedef struct {
    int x, y, v;
} Event;

static Event events[MAXE];
static int ev_cnt;

static int xs[MAXX], xlen;
static int ys[MAXY], ylen;

static int ev_start[MAXX], ev_end[MAXX];
static int line_map[MAXY];

static int cmp_int(const void *a, const void *b) {
    return (*(int*)a) - (*(int*)b);
}

static int find_idx(int *arr, int n, int val) {
    int l = 0, r = n - 1;
    while (l <= r) {
        int m = (l + r) / 2;
        if (arr[m] < val) l = m + 1;
        else if (arr[m] > val) r = m - 1;
        else return m;
    }
    return -1;
}

static void build_events_and_compress(int **points, int n, int day) {
    ev_cnt = 0;
    for (int i = 0; i < n; i++) {
        int x = points[i][0], y = points[i][1];
        int a = x + y - day;
        int b = y - x - day;
        int c = y - x + day + 1;
        int d = x + y + day + 1;
        events[ev_cnt++] = (Event){a, b, +1};
        events[ev_cnt++] = (Event){a, c, -1};
        events[ev_cnt++] = (Event){d, b, -1};
        events[ev_cnt++] = (Event){d, c, +1};
    }

    xlen = ylen = 0;
    for (int i = 0; i < ev_cnt; i++) {
        xs[xlen++] = events[i].x;
        ys[ylen++] = events[i].y;
    }
    qsort(xs, xlen, sizeof(int), cmp_int);
    qsort(ys, ylen, sizeof(int), cmp_int);
    int ux = 1, uy = 1;
    for (int i = 1; i < xlen; i++)
        if (xs[i] != xs[i-1]) xs[ux++] = xs[i];
    for (int i = 1; i < ylen; i++)
        if (ys[i] != ys[i-1]) ys[uy++] = ys[i];
    xlen = ux;  ylen = uy;

    for (int i = 0; i < xlen; i++) ev_start[i] = ev_end[i] = 0;
    for (int i = 0; i < ev_cnt; i++) {
        int xi = find_idx(xs, xlen, events[i].x);
        ev_end[xi]++;
    }
    for (int i = 1; i < xlen; i++)
        ev_end[i] += ev_end[i-1];
    for (int i = xlen-1; i > 0; i--)
        ev_start[i] = ev_end[i-1];
    ev_start[0] = 0;
    Event tmp[MAXE];
    memcpy(tmp, events, ev_cnt * sizeof(Event));
    int ptr[MAXX];
    memcpy(ptr, ev_start, xlen * sizeof(int));
    for (int i = 0; i < ev_cnt; i++) {
        int xi = find_idx(xs, xlen, tmp[i].x);
        events[ ptr[xi]++ ] = tmp[i];
    }
}

static int check(int **points, int n, int k, int day) {
    build_events_and_compress(points, n, day);

    memset(line_map, 0, ylen * sizeof(int));

    int base = 0;
    for (int xi = 0; xi < xlen; xi++) {
        int lo = (xi==0 ? 0 : ev_end[xi-1]);
        int hi = ev_end[xi];
        int xx = xs[xi];
        for (int e = lo; e < hi; e++) {
            int yi = find_idx(ys, ylen, events[e].y);
            line_map[yi] += events[e].v;
        }
        int prefix = 0;
        for (int yj = 0; yj < ylen; yj++) {
            prefix += line_map[yj];
            if (prefix < k) continue;
            int yy = ys[yj];
            if (((xx - yy) & 1) == 0) return 1;
            if (find_idx(xs, xlen, xx + 1) < 0) return 1;
            int j1 = find_idx(ys, ylen, yy + 1);
            int nxt = (j1 >= 0 ? line_map[j1] : 0);
            if (prefix + nxt >= k) return 1;
        }
    }
    return 0;
}

int minDayskVariants(int **points, int pointsSize, int *pointsColSize, int k) {
    int lo = 0, hi = 1000000000;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (check(points, pointsSize, k, mid))
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}
