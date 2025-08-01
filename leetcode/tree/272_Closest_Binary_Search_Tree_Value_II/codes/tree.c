#include <stdlib.h>
#include <math.h>

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

void inorder(struct TreeNode* root, int* values, int* idx) {
    if (!root) return;
    inorder(root -> left, values, idx);
    values[(*idx)++] = root -> val;
    inorder(root -> right, values, idx);
}

int compare(const void* a, const void* b, void* target_ptr) {
    double target = *(double*)target_ptr;
    double diff1 = fabs(*(int*)a - target);
    double diff2 = fabs(*(int*)b - target);
    if (diff1 < diff2) return -1;
    else if (diff1 > diff2) return 1;
    else return *(int*)a - *(int*)b;  
}

int* closestKValues(struct TreeNode* root, double target, int k, int* returnSize) {
    int* all_values = malloc(sizeof(int) * 10000);
    int idx = 0;
    inorder(root, all_values, &idx);

#if defined(_GNU_SOURCE)
    qsort_r(all_values, idx, sizeof(int), compare, &target);
#else
    static double g_target;
    g_target = target;
    int cmp(const void* a, const void* b) {
        double diff1 = fabs(*(int*)a - g_target);
        double diff2 = fabs(*(int*)b - g_target);
        if (diff1 < diff2) return -1;
        else if (diff1 > diff2) return 1;
        else return *(int*)a - *(int*)b;
    }
    qsort(all_values, idx, sizeof(int), cmp);
#endif

    int* result = malloc(sizeof(int) * k);
    for (int i = 0; i < k; i++) {
        result[i] = all_values[i];
    }

    *returnSize = k;
    free(all_values);
    return result;
}
