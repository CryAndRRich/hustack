#include<unordered_set>

using namespace std;

class Node {
public:
    int val;
    Node* left;
    Node* right;
    Node* parent;
};

class Solution {
public:
    Node* lowestCommonAncestor(Node* p, Node* q) {
        unordered_set<Node*> ancestors;

        while (p) {
            ancestors.insert(p);
            p = p -> parent;
        }

        while (q) {
            if (ancestors.count(q)) {
                return q;
            }
            q = q -> parent;
        }

        return nullptr;
    }
};
