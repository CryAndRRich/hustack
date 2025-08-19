/**
Definition for a binary tree node.
 */
struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

class Solution {
public:
    int heightOfTree(TreeNode* root) {
        unordered_map<int, int> countMap;
        dfs(root, countMap);
        return maxDepth(root, 0, countMap);
    }

private:
    void dfs(TreeNode* node, unordered_map<int, int>& countMap) {
        if (!node) return;
        countMap[node -> val]++;
        if (countMap[node -> val] > 1) return;
        dfs(node -> left, countMap);
        dfs(node -> right, countMap);
    }

    int maxDepth(TreeNode* node, int depth, unordered_map<int, int>& countMap) {
        if (!node || countMap[node -> val] > 1) return depth;
        int left = maxDepth(node -> left, depth + 1, countMap);
        int right = maxDepth(node -> right, depth + 1, countMap);
        return max(left, right);
    }
};
