#include <stdlib.h>
#include <math.h>
#define MAXN 10000

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

int absDiff(double a, double b) {
    return fabs(a - b);
}

void push(struct TreeNode** stack, int* size, struct TreeNode* node) {
    stack[(*size)++] = node;
}

struct TreeNode* pop(struct TreeNode** stack, int* size) {
    return stack[--(*size)];
}

void initPred(struct TreeNode* root, double target, struct TreeNode** stack, int* size) {
    while (root) {
        if (root -> val <= target) {
            push(stack, size, root);
            root = root -> right;
        } else {
            root = root -> left;
        }
    }
}

void initSucc(struct TreeNode* root, double target, struct TreeNode** stack, int* size) {
    while (root) {
        if (root -> val > target) {
            push(stack, size, root);
            root = root -> left;
        } else {
            root = root -> right;
        }
    }
}

int getNextPred(struct TreeNode** stack, int* size) {
    struct TreeNode* node = pop(stack, size);
    int val = node -> val;
    node = node -> left;
    while (node) {
        push(stack, size, node);
        node = node -> right;
    }
    return val;
}

int getNextSucc(struct TreeNode** stack, int* size) {
    struct TreeNode* node = pop(stack, size);
    int val = node -> val;
    node = node -> right;
    while (node) {
        push(stack, size, node);
        node = node -> left;
    }
    return val;
}

int* closestKValues(struct TreeNode* root, double target, int k, int* returnSize) {
    struct TreeNode* pred[MAXN], *succ[MAXN];
    int ps = 0, ss = 0;
    initPred(root, target, pred, &ps);
    initSucc(root, target, succ, &ss);

    int* result = malloc(sizeof(int) * k);
    int ri = 0;
    while (k--) {
        if (ps == 0) {
            result[ri++] = getNextSucc(succ, &ss);
        } else if (ss == 0) {
            result[ri++] = getNextPred(pred, &ps);
        } else if (fabs(pred[ps - 1] -> val - target) <= fabs(succ[ss - 1] -> val - target)) {
            result[ri++] = getNextPred(pred, &ps);
        } else {
            result[ri++] = getNextSucc(succ, &ss);
        }
    }
    *returnSize = ri;
    return result;
}
