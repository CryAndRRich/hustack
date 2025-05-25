#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

typedef struct {
    int x, y;
} Point;

int compareX(const void* a, const void* b) {
    return ((Point*)a) -> x - ((Point*)b) -> x;
}

int compareY(const void* a, const void* b) {
    return ((Point*)a) -> y - ((Point*)b) -> y;
}

double dist(Point a, Point b) {
    double dx = (double)a.x - b.x;
    double dy = (double)a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

double bruteForce(Point* P, int n) {
    double minD = DBL_MAX;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (dist(P[i], P[j]) < minD)
                minD = dist(P[i], P[j]);
    return minD;
}

double stripClosest(Point* strip, int size, double d) {
    double minD = d;
    qsort(strip, size, sizeof(Point), compareY);
    for (int i = 0; i < size; ++i)
        for (int j = i + 1; j < size && (strip[j].y - strip[i].y) < minD; ++j)
            if (dist(strip[i], strip[j]) < minD)
                minD = dist(strip[i], strip[j]);
    return minD;
}

double closestUtil(Point* Px, int n) {
    if (n <= 3) return bruteForce(Px, n);

    int mid = n / 2;
    Point midPoint = Px[mid];

    double dl = closestUtil(Px, mid);
    double dr = closestUtil(Px + mid, n - mid);
    double d = fmin(dl, dr);

    Point* strip = (Point*)malloc(n * sizeof(Point));
    int j = 0;
    for (int i = 0; i < n; i++)
        if (fabs(Px[i].x - midPoint.x) < d)
            strip[j++] = Px[i];

    double minD = fmin(d, stripClosest(strip, j, d));
    free(strip);
    return minD;
}

double closest(Point* P, int n) {
    qsort(P, n, sizeof(Point), compareX);
    return closestUtil(P, n);
}

int main() {
    int n;
    scanf("%d", &n);
    Point* points = (Point*)malloc(n * sizeof(Point));
    for (int i = 0; i < n; ++i)
        scanf("%d %d", &points[i].x, &points[i].y);

    double res = closest(points, n);
    printf("%.6f", res);
    free(points);
    return 0;
}
