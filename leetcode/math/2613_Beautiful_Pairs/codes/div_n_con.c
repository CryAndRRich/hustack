#include <stdlib.h>
#include <limits.h>
#include <math.h>

typedef struct { int x, y, idx; } Point;

static long long hash_xy(int x, int y) {
    return ((long long)x << 32) | (unsigned int)y;
}

static int cmp_points(const void *a, const void *b) {
    Point *pa = (Point *)a, *pb = (Point *)b;
    if (pa -> x != pb -> x) return pa -> x - pb -> x;
    if (pa -> y != pb -> y) return pa -> y - pb -> y;
    return pa -> idx - pb -> idx;
}

static int cmp_y(const void *a, const void *b) {
    Point *pa = (Point *)a, *pb = (Point *)b;
    if (pa -> y != pb -> y) return pa -> y - pb -> y;
    return pa -> idx - pb -> idx;
}

static void closest(Point *points, int n, int *mind, int *resi, int *resj);
static void strip_closest(Point *strip, int m, int d, int *mind, int *si, int *sj);

int* beautifulPair(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize) {
    int n = nums1Size;
    *returnSize = 2;
    int *res = (int*)malloc(sizeof(int) * 2);

    typedef struct { long long key; int val; } Entry;
    Entry *table = (Entry*)malloc(sizeof(Entry) * n);
    int table_size = 0;
    int best_i = INT_MAX, best_j = INT_MAX;

    for (int i = 0; i < n; i++) {
        long long h = hash_xy(nums1[i], nums2[i]);
        int found = 0;
        for (int k = 0; k < table_size; k++) {
            if (table[k].key == h) {
                int ci = table[k].val, cj = i;
                if (ci < best_i || (ci == best_i && cj < best_j)) {
                    best_i = ci; best_j = cj;
                }
                found = 1; break;
            }
        }
        if (!found) {
            table[table_size].key = h;
            table[table_size].val = i;
            table_size++;
        }
    }
    free(table);

    if (best_i != INT_MAX) {
        res[0] = best_i; res[1] = best_j;
        return res;
    }

    Point *points = (Point*)malloc(sizeof(Point) * n);
    for (int i = 0; i < n; i++) {
        points[i].x = nums1[i];
        points[i].y = nums2[i];
        points[i].idx = i;
    }
    qsort(points, n, sizeof(Point), cmp_points);

    int mind = INT_MAX, resi, resj;
    closest(points, n, &mind, &resi, &resj);
    if (resi > resj) { int tmp = resi; resi = resj; resj = tmp; }
    res[0] = resi; res[1] = resj;
    free(points);
    return res;
}

static void closest(Point *points, int n, int *mind, int *resi, int *resj) {
    if (n < 2) { *mind = INT_MAX; *resi = *resj = INT_MAX; return; }
    int mid = n / 2;
    Point midP = points[mid];

    int dl, pi, pj;
    closest(points, mid, &dl, &pi, &pj);
    int dr, qi, qj;
    closest(points + mid, n - mid, &dr, &qi, &qj);

    *mind = dl; *resi = pi; *resj = pj;
    if (dr < *mind || (dr == *mind && (qi < *resi || (qi == *resi && qj < *resj)))) {
        *mind = dr; *resi = qi; *resj = qj;
    }
    int d = *mind;

    Point *strip = (Point*)malloc(sizeof(Point) * n);
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (abs(points[i].x - midP.x) <= d) strip[m++] = points[i];
    }
    strip_closest(strip, m, d, mind, resi, resj);
    free(strip);
}

static void strip_closest(Point *strip, int m, int d, int *mind, int *si, int *sj) {
    qsort(strip, m, sizeof(Point), cmp_y);
    for (int i = 0; i < m; i++) {
        for (int j = i+1; j < m; j++) {
            if (strip[j].y - strip[i].y > *mind) break;
            int dist = abs(strip[j].x - strip[i].x) + abs(strip[j].y - strip[i].y);
            int ci = strip[i].idx, cj = strip[j].idx;
            if (ci > cj) { int tmp = ci; ci = cj; cj = tmp; }
            if (dist < *mind || (dist == *mind && (ci < *si || (ci == *si && cj < *sj)))) {
                *mind = dist; *si = ci; *sj = cj;
            }
        }
    }
}
