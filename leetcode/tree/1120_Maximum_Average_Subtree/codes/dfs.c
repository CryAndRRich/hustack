/**
Definition for a binary tree node.
};
 */
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;

struct Result {
    int sum;
    int count;
};

double maxAvg = 0;

struct Result dfs(struct TreeNode* node) {
    if (!node) return (struct Result){0, 0};

    struct Result left = dfs(node -> left);
    struct Result right = dfs(node -> right);

    int totalSum = left.sum + right.sum + node -> val;
    int totalCount = left.count + right.count + 1;

    double avg = (double)totalSum / totalCount;
    if (avg > maxAvg) maxAvg = avg;

    return (struct Result){totalSum, totalCount};
}

double maximumAverageSubtree(struct TreeNode* root) {
    maxAvg = 0;
    dfs(root);
    return maxAvg;
}
