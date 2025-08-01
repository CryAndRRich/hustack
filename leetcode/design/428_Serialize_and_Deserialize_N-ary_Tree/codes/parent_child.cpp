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
        if (!root) return "";
        string result;
        serializeHelper(root, result);
        return result;
    }

    void serializeHelper(Node* node, string& result) {
        if (!node) return;
        result += to_string(node -> val) + " ";
        result += to_string(node -> children.size()) + " ";
        for (Node* child : node -> children) {
            serializeHelper(child, result);
        }
    }

    Node* deserialize(string data) {
        if (data.empty()) return nullptr;
        istringstream iss(data);
        return deserializeHelper(iss);
    }

    Node* deserializeHelper(istringstream& iss) {
        int val, size;
        iss >> val >> size;
        Node* node = new Node(val);
        for (int i = 0; i < size; ++i) {
            node -> children.push_back(deserializeHelper(iss));
        }
        return node;
    }
};

// Your Codec object will be instantiated and called as such:
// Codec codec;
// codec.deserialize(codec.serialize(root));