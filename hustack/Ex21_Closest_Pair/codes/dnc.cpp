#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

struct Point {
    int x, y;
};

bool compareX(const Point& a, const Point& b) {
    return a.x < b.x;
}

bool compareY(const Point& a, const Point& b) {
    return a.y < b.y;
}

double dist(const Point& a, const Point& b) {
    double dx = (double)a.x - b.x;
    double dy = (double)a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

double bruteForce(const vector<Point>& P, int left, int right) {
    double minD = numeric_limits<double>::max();
    for (int i = left; i <= right; ++i)
        for (int j = i + 1; j <= right; ++j) {
            double d = dist(P[i], P[j]);
            if (d < minD) minD = d;
        }
    return minD;
}

double stripClosest(vector<Point>& strip, double d) {
    double minD = d;
    sort(strip.begin(), strip.end(), compareY);

    int size = strip.size();
    for (int i = 0; i < size; ++i)
        for (int j = i + 1; j < size && (strip[j].y - strip[i].y) < minD; ++j) {
            double d_ij = dist(strip[i], strip[j]);
            if (d_ij < minD)
                minD = d_ij;
        }
    return minD;
}

double closestUtil(vector<Point>& Px, int left, int right) {
    if (right - left + 1 <= 3)
        return bruteForce(Px, left, right);

    int mid = (left + right) / 2;
    Point midPoint = Px[mid];

    double dl = closestUtil(Px, left, mid);
    double dr = closestUtil(Px, mid + 1, right);
    double d = min(dl, dr);

    vector<Point> strip;
    for (int i = left; i <= right; ++i)
        if (fabs(Px[i].x - midPoint.x) < d)
            strip.push_back(Px[i]);

    return min(d, stripClosest(strip, d));
}

double closest(vector<Point>& P) {
    sort(P.begin(), P.end(), compareX);
    return closestUtil(P, 0, (int)P.size() - 1);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; 
    cin >> n;
    vector<Point> points(n);
    for (int i = 0; i < n; ++i)
        cin >> points[i].x >> points[i].y;

    double res = closest(points);
    cout.precision(6);
    cout << fixed << res << "\n";

    return 0;
}
