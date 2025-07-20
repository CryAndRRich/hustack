#include <vector>
#include <array>
#include <algorithm>
#include <climits>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int v) : val(v), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    array<int, 2> dfs(TreeNode* node) {
        if (!node) return {INT_MAX, INT_MAX};

        if (!node -> left && !node -> right) {
            int val = node -> val;
            return val == 0 ? array<int, 2>{0, 1} : array<int, 2>{1, 0};  
        }

        if (node -> val == 5) { 
            TreeNode* child = node -> left ? node -> left : node -> right;
            auto c = dfs(child);
            return {c[1], c[0]};
        }

        auto l = dfs(node -> left);
        auto r = dfs(node -> right);
        array<int, 2> dp = {INT_MAX, INT_MAX};

        for (int lv = 0; lv <= 1; lv++) {
            for (int rv = 0; rv <= 1; rv++) {
                int res = 0;
                if (node -> val == 2) res = lv | rv;
                else if (node -> val == 3) res = lv & rv;
                else if (node -> val == 4) res = lv ^ rv;

                dp[res] = min(dp[res], l[lv] + r[rv]);
            }
        }

        return dp;
    }

    int minimumFlips(TreeNode* root, bool result) {
        auto dp = dfs(root);
        return result ? dp[1] : dp[0];
    }
};
