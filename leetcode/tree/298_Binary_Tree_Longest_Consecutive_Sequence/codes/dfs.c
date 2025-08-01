/**
Definition for a binary tree node.
 */
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

void dfs(struct TreeNode* node, int prevVal, int currLen, int* maxLen) {
    if (!node) return;
    if (node -> val == prevVal + 1)
        currLen++;
    else
        currLen = 1;
    if (currLen > *maxLen)
        *maxLen = currLen;
    dfs(node -> left, node -> val, currLen, maxLen);
    dfs(node -> right, node -> val, currLen, maxLen);
}

int longestConsecutive(struct TreeNode* root) {
    int maxLen = 0;
    dfs(root, root -> val - 1, 0, &maxLen);
    return maxLen;
}
