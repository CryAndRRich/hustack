/**
Definition for a binary tree node.
 */

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

bool isLeaf(struct TreeNode* node) {
    return node && !node -> left && !node -> right;
}

void addLeftBoundary(struct TreeNode* node, int* res, int* size) {
    while (node) {
        if (!isLeaf(node)) res[(*size)++] = node -> val;
        node = node -> left ? node -> left : node -> right;
    }
}

void addRightBoundary(struct TreeNode* node, int* stack, int* top) {
    while (node) {
        if (!isLeaf(node)) stack[(*top)++] = node -> val;
        node = node -> right ? node -> right : node -> left;
    }
}

void addLeaves(struct TreeNode* node, int* res, int* size) {
    if (!node) return;
    if (isLeaf(node)) {
        res[(*size)++] = node -> val;
        return;
    }
    addLeaves(node -> left, res, size);
    addLeaves(node -> right, res, size);
}

/**
Note: The returned array must be malloced, assume caller calls free().
 */
int* boundaryOfBinaryTree(struct TreeNode* root, int* returnSize){
    int* res = malloc(10000sizeof(int));
    int* rightStack = malloc(10000sizeof(int));
    int size = 0, top = 0;

    if (!root) {
        *returnSize = 0;
        return res;
    }

    if (!isLeaf(root)) res[size++] = root -> val;
    addLeftBoundary(root -> left, res, &size);
    addLeaves(root, res, &size);
    addRightBoundary(root -> right, rightStack, &top);

    for (int i = top - 1; i >= 0; i--)
        res[size++] = rightStack[i];

    *returnSize = size;
    free(rightStack);
    return res;
}
