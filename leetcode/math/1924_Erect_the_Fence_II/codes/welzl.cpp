#include <bits/stdc++.h>
using namespace std;

struct Point { double x, y; };

double dist(const Point &a, const Point &b) {
    return hypot(a.x - b.x, a.y - b.y);
}
Point mid(const Point &a, const Point &b) {
    return {(a.x + b.x) / 2.0, (a.y + b.y) / 2.0};
}
pair<Point,double> circleFrom(vector<Point> R) {
    if (R.empty()) return {{0,0}, 0};
    if (R.size() == 1) return {R[0], 0};
    if (R.size() == 2) {
        Point c = mid(R[0], R[1]);
        return {c, dist(R[0], R[1]) / 2.0};
    }
    Point A = R[0], B = R[1], C = R[2];
    double a = B.x - A.x, b = B.y - A.y;
    double c = C.x - A.x, d = C.y - A.y;
    double e = a * (A.x + B.x) + b * (A.y + B.y);
    double f = c * (A.x + C.x) + d * (A.y + C.y);
    double g = 2 * (a * (C.y - B.y) - b * (C.x - B.x));
    if (fabs(g) < 1e-12) return {{0,0}, numeric_limits<double>::infinity()};
    double cx = (d * e - b * f) / g;
    double cy = (a * f - c * e) / g;
    return {{cx, cy}, dist({cx, cy}, A)};
}
bool inCircle(Point p, Point c, double r) {
    return dist(p, c) <= r + 1e-12;
}
pair<Point,double> welzl(vector<Point> &P, vector<Point> R, int n) {
    if (n == 0 || R.size() == 3) return circleFrom(R);
    int idx = rand() % n;
    swap(P[idx], P[n - 1]);
    Point p = P[n - 1];
    auto circle = welzl(P, R, n - 1);
    if (inCircle(p, circle.first, circle.second)) return circle;
    R.push_back(p);
    return welzl(P, R, n - 1);
}
class Solution {
public:
    vector<double> outerTrees(vector<vector<int>>& trees) {
        set<pair<int,int>> s;
        vector<Point> pts;
        for (auto &t : trees) {
            if (s.insert({t[0], t[1]}).second) pts.push_back({(double)t[0], (double)t[1]});
        }
        random_shuffle(pts.begin(), pts.end());
        auto res = welzl(pts, {}, pts.size());
        return {res.first.x, res.first.y, res.second};
    }
};
