#include <bits/stdc++.h>
using namespace std;

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
    pair<priority_queue<int>, int> dfs(TreeNode* root, int k) {
        if (!root) return {{}, 0};

        auto left = dfs(root->left, k);
        auto right = dfs(root->right, k);

        while (!right.first.empty()) {
            left.first.push(right.first.top());
            right.first.pop();
            if ((int)left.first.size() > k) left.first.pop();
        }

        int count = left.second + right.second;
        if ((int)left.first.size() == k && left.first.top() < root->val) count++;

        left.first.push(root->val);
        if ((int)left.first.size() > k) left.first.pop();

        return {left.first, count};
    }
public:
    int countGreatEnoughNodes(TreeNode* root, int k) {
        return dfs(root, k).second;
    }
};
