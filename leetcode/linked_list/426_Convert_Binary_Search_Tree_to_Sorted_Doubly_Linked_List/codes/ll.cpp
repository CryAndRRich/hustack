// Definition for a Node.
class Node {
public:
    int val;
    Node* left;
    Node* right;

    Node() {}
    Node(int _val) {
        val = _val;
        left = nullptr;
        right = nullptr;
    }
    Node(int _val, Node* _left, Node* _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};

class Solution {
public:
    Node* first = nullptr;
    Node* last = nullptr;

    Node* treeToDoublyList(Node* root) {
        if (!root) return nullptr;
        dfs(root);
        first -> left = last;
        last -> right = first;
        return first;
    }

    void dfs(Node* node) {
        if (!node) return;
        dfs(node -> left);
        if (last) {
            last -> right = node;
            node -> left = last;
        } else {
            first = node;
        }
        last = node;
        dfs(node -> right);
    }
};
