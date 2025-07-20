#include <stdbool.h>

#define MAX_NODES 10001

typedef struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

bool isTarget[MAX_NODES] = {0};

TreeNode* dfs(TreeNode* node) {
    if (!node) return NULL;
    if (isTarget[node -> val + 100000]) return node;
    TreeNode* left = dfs(node -> left);
    TreeNode* right = dfs(node -> right);
    if (left && right) return node;
    return left ? left : right;
}

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode** nodes, int nodesSize) {
    for (int i = 0; i < nodesSize; i++) {
        isTarget[nodes[i] -> val + 100000] = true;
    }
    return dfs(root);
}
