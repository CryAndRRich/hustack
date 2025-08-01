/**
Definition for a binary tree node.
 */
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

bool isUnival(struct TreeNode* node, int* count) {
    if (!node) return true;

    bool left = isUnival(node->left, count);
    bool right = isUnival(node->right, count);

    if (!left || !right) return false;

    if (node->left && node->left->val != node->val) return false;
    if (node->right && node->right->val != node->val) return false;

    (*count)++;
    return true;
}

int countUnivalSubtrees(struct TreeNode* root) {
    int count = 0;
    isUnival(root, &count);
    return count;
}
