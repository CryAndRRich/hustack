#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <limits>

using namespace std;

struct Point {
    int x, y;
};

struct Node {
    Point pt;
    unique_ptr<Node> left;
    unique_ptr<Node> right;
    int depth;
    Node(Point p, int d) : pt(p), depth(d), left(nullptr), right(nullptr) {}
};

unique_ptr<Node> insert(unique_ptr<Node> root, const Point& pt, int depth) {
    if (!root) {
        return make_unique<Node>(pt, depth);
    }
    int cd = depth % 2;
    if ((cd == 0 && pt.x < root -> pt.x) || (cd == 1 && pt.y < root -> pt.y))
        root -> left = insert(move(root -> left), pt, depth + 1);
    else
        root -> right = insert(move(root -> right), pt, depth + 1);
    return root;
}

double dist(const Point& a, const Point& b) {
    double dx = (double)a.x - b.x;
    double dy = (double)a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

void nearest(const unique_ptr<Node>& root, const Point& target, double& bestDist) {
    if (!root) return;

    double d = dist(root -> pt, target);
    if (d > 0 && d < bestDist)
        bestDist = d;

    int cd = root -> depth % 2;
    const unique_ptr<Node>* first = nullptr;
    const unique_ptr<Node>* second = nullptr;

    if ((cd == 0 && target.x < root -> pt.x) || (cd == 1 && target.y < root -> pt.y)) {
        first = &root -> left;
        second = &root -> right;
    } else {
        first = &root -> right;
        second = &root -> left;
    }

    nearest(*first, target, bestDist);

    double delta = (cd == 0) ? fabs(target.x - root -> pt.x) : fabs(target.y - root -> pt.y);
    if (delta < bestDist)
        nearest(*second, target, bestDist);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    vector<Point> points(n);
    for (int i = 0; i < n; ++i) {
        cin >> points[i].x >> points[i].y;
    }

    unique_ptr<Node> root = nullptr;
    double minD = numeric_limits<double>::max();

    for (int i = 0; i < n; ++i) {
        double d = numeric_limits<double>::max();
        nearest(root, points[i], d);
        if (d < minD) minD = d;
        root = insert(move(root), points[i], 0);
    }

    cout.precision(6);
    cout << fixed << minD << "\n";

    return 0;
}
