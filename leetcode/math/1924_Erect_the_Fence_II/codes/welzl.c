#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

typedef struct { double x, y; } Point;

double dist(Point a, Point b) {
    return hypot(a.x - b.x, a.y - b.y);
}
Point mid(Point a, Point b) {
    Point m = {(a.x + b.x) / 2.0, (a.y + b.y) / 2.0};
    return m;
}
void circleFrom(Point* R, int rSize, Point* center, double *  radius) {
    if (rSize == 0) { 
        center -> x = 0; 
        center -> y = 0;  
        *radius = 0; 
        return; 
    }
    if (rSize == 1) { 
        *center = R[0];  
        *radius = 0; 
        return; 
    }
    if (rSize == 2) { 
        *center = mid(R[0], R[1]);  
        *radius = dist(R[0], R[1]) / 2.0; 
        return; 
    }
    Point A = R[0], B = R[1], C = R[2];
    double a = B.x - A.x, b = B.y - A.y;
    double c = C.x - A.x, d = C.y - A.y;
    double e = a * (A.x + B.x) + b * (A.y + B.y);
    double f = c * (A.x + C.x) + d * (A.y + C.y);
    double g = 2 * (a * (C.y - B.y) - b * (C.x - B.x));

    if (fabs(g) < 1e-12) { 
        center -> x = 0; center -> y = 0;  
        *radius = INFINITY; return; 
    }
    center -> x = (d * e - b * f) / g;
    center -> y = (a * f - c * e) / g;
    *radius = dist(*center, A);
}
bool inCircle(Point p, Point c, double r) {
    return dist(p, c) <= r + 1e-12;
}
void welzl(Point *  P, int n, Point *  R, int rSize, Point *  center, double *  radius) {
    if (n == 0 || rSize == 3) { circleFrom(R, rSize, center, radius); return; }
    int idx = rand() % n;
    Point temp = P[idx]; P[idx] = P[n - 1]; P[n - 1] = temp;
    Point p = P[n - 1];
    Point c1; double r1;
    welzl(P, n - 1, R, rSize, &c1, &r1);
    if (inCircle(p, c1, r1)) {  * center = c1;  * radius = r1; return; }
    R[rSize] = p;
    welzl(P, n - 1, R, rSize + 1, center, radius);
}
double *  outerTrees(int *  *  trees, int treesSize, int *  treesColSize, int *  returnSize) {
    Point *  pts = (Point * )malloc(treesSize  *  sizeof(Point));
    int m = 0;
    for (int i = 0; i < treesSize; i++) {
        int unique = 1;
        for (int j = 0; j < m; j++) {
            if ((int)pts[j].x == trees[i][0] && (int)pts[j].y == trees[i][1]) { unique = 0; break; }
        }
        if (unique) { pts[m].x = trees[i][0]; pts[m].y = trees[i][1]; m++; }
    }
    for (int i = m - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        Point tmp = pts[i]; pts[i] = pts[j]; pts[j] = tmp;
    }
    Point R[3]; Point center; 
    double radius;
    welzl(pts, m, R, 0, &center, &radius);
    double *  ans = (double * )malloc(3  *  sizeof(double));
    ans[0] = center.x; ans[1] = center.y; ans[2] = radius;
     * returnSize = 3;
    free(pts);
    return ans;
}
