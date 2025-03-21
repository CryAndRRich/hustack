#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    double x, y;
    int axis; 
    struct Node* left;
    struct Node* right;
} Node;

int cmpX(const void *a, const void *b) {
    double diff = ((Node*)a)->x - ((Node*)b)->x;
    return (diff < 0) ? -1 : (diff > 0) ? 1 : 0;
}

int cmpY(const void *a, const void *b) {
    double diff = ((Node*)a)->y - ((Node*)b)->y;
    return (diff < 0) ? -1 : (diff > 0) ? 1 : 0;
}

Node* buildKdTree(Node* nodes, int n, int depth) {
    if(n <= 0)
        return NULL;
    
    int axis = depth % 2; 
    if(axis == 0)
        qsort(nodes, n, sizeof(Node), cmpX);
    else
        qsort(nodes, n, sizeof(Node), cmpY);
    
    int mid = n / 2;
    Node* root = (Node*) malloc(sizeof(Node));
    root->x = nodes[mid].x;
    root->y = nodes[mid].y;
    root->axis = axis;
    
    root->left = buildKdTree(nodes, mid, depth + 1);
    root->right = buildKdTree(nodes + mid + 1, n - mid - 1, depth + 1);
    
    return root;
}

int rangeQuery(Node* root, double left, double top, double right, double bottom) {
    if(root == NULL)
        return 0;
    
    int count = 0;
    if(root->x > left && root->x < right && root->y < top && root->y > bottom)
        count++;
    
    if(root->axis == 0) {
        if(left < root->x)
            count += rangeQuery(root->left, left, top, right, bottom);
        if(root->x < right)
            count += rangeQuery(root->right, left, top, right, bottom);
    } else {
        if(root->y > bottom)
            count += rangeQuery(root->left, left, top, right, bottom);
        if(root->y < top)
            count += rangeQuery(root->right, left, top, right, bottom);
    }
    
    return count;
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    
    Node* nodes = (Node*) malloc(n * sizeof(Node));
    for (int i = 0; i < n; i++) {
        scanf("%lf %lf", &nodes[i].x, &nodes[i].y);
    }
    
    Node* root = buildKdTree(nodes, n, 0);
    
    for (int q = 0; q < m; q++){
        double left, top, right, bottom;
        scanf("%lf %lf %lf %lf", &left, &top, &right, &bottom);
        int count = rangeQuery(root, left, top, right, bottom);
        printf("%d\n", count);
    }
    
    free(nodes);
    return 0;
}
