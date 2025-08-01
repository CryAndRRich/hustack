#include <string>
#include <vector>
#include <sstream>
using namespace std;

// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};

class Codec {
public:
    string serialize(Node* root) {
        stringstream ss;
        serializeHelper(root, ss);
        return ss.str();
    }

    void serializeHelper(Node* root, stringstream& ss) {
        if (!root) return;
        ss << root -> val << " " << root -> children.size() << " ";
        for (auto child : root -> children) {
            serializeHelper(child, ss);
        }
    }

    Node* deserialize(string data) {
        if (data.empty()) return nullptr;
        stringstream ss(data);
        return deserializeHelper(ss);
    }

    Node* deserializeHelper(stringstream& ss) {
        int val, size;
        if (!(ss >> val)) return nullptr;
        ss >> size;
        Node* node = new Node(val);
        for (int i = 0; i < size; ++i) {
            node -> children.push_back(deserializeHelper(ss));
        }
        return node;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec codec;
// codec.deserialize(codec.serialize(root));