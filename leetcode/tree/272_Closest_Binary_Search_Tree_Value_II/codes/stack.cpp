#include <vector>
#include <stack>
#include <cmath>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    vector<int> closestKValues(TreeNode* root, double target, int k) {
        vector<int> res;
        stack<TreeNode*> pred, succ;

        auto initPredecessor = [&](TreeNode* node) {
            while (node) {
                if (node -> val <= target) {
                    pred.push(node);
                    node = node -> right;
                } else {
                    node = node -> left;
                }
            }
        };

        auto initSuccessor = [&](TreeNode* node) {
            while (node) {
                if (node -> val > target) {
                    succ.push(node);
                    node = node -> left;
                } else {
                    node = node -> right;
                }
            }
        };

        auto getNextPred = [&]() {
            TreeNode* node = pred.top(); pred.pop();
            int val = node -> val;
            node = node -> left;
            while (node) {
                pred.push(node);
                node = node -> right;
            }
            return val;
        };

        auto getNextSucc = [&]() {
            TreeNode* node = succ.top(); succ.pop();
            int val = node -> val;
            node = node -> right;
            while (node) {
                succ.push(node);
                node = node -> left;
            }
            return val;
        };

        initPredecessor(root);
        initSuccessor(root);

        for (int i = 0; i < k; ++i) {
            if (pred.empty()) {
                res.push_back(getNextSucc());
            } else if (succ.empty()) {
                res.push_back(getNextPred());
            } else if (abs(pred.top() -> val - target) <= abs(succ.top() -> val - target)) {
                res.push_back(getNextPred());
            } else {
                res.push_back(getNextSucc());
            }
        }

        return res;
    }
};
