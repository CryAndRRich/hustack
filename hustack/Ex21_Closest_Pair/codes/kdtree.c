#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

typedef struct {
    int x, y;
} Point;

typedef struct Node {
    Point pt;
    struct Node* left;
    struct Node* right;
    int depth;
} Node;

Node* insert(Node* root, Point pt, int depth) {
    if (root == NULL) {
        Node* node = (Node*)malloc(sizeof(Node));
        node -> pt = pt;
        node -> left = node -> right = NULL;
        node -> depth = depth;
        return node;
    }

    int cd = depth % 2;
    if ((cd == 0 && pt.x < root -> pt.x) || (cd == 1 && pt.y < root -> pt.y))
        root -> left = insert(root -> left, pt, depth + 1);
    else
        root -> right = insert(root -> right, pt, depth + 1);
    return root;
}

double dist(Point a, Point b) {
    double dx = (double)a.x - b.x;
    double dy = (double)a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

void nearest(Node* root, Point target, double* bestDist) {
    if (root == NULL) return;

    double d = dist(root -> pt, target);
    if (d > 0 && d < *bestDist)
        *bestDist = d;

    int cd = root -> depth % 2;
    Node* first = NULL, * second = NULL;

    if ((cd == 0 && target.x < root -> pt.x) || (cd == 1 && target.y < root -> pt.y)) {
        first = root -> left;
        second = root -> right;
    } else {
        first = root -> right;
        second = root -> left;
    }

    nearest(first, target, bestDist);

    double delta = cd == 0 ? fabs(target.x - root -> pt.x) : fabs(target.y - root -> pt.y);
    if (delta < *bestDist)
        nearest(second, target, bestDist);
}

int main() {
    int n;
    scanf("%d", &n);
    Point* points = (Point*)malloc(n * sizeof(Point));
    for (int i = 0; i < n; i++)
        scanf("%d %d", &points[i].x, &points[i].y);

    Node* root = NULL;
    double minD = DBL_MAX;
    for (int i = 0; i < n; i++) {
        double d = DBL_MAX;
        nearest(root, points[i], &d);
        if (d < minD) minD = d;
        root = insert(root, points[i], 0);
    }

    printf("%.6f", minD);
    free(points);
    return 0;
}
