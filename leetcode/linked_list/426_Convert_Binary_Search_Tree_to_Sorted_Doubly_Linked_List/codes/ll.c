// Definition for a Node
struct Node {
    int val;
    struct Node* left;
    struct Node* right;
};

struct Node* firstNode = NULL;
struct Node* lastNode = NULL;

void dfs(struct Node* node) {
    if (!node) return;
    dfs(node -> left);
    if (lastNode) {
        lastNode -> right = node;
        node -> left = lastNode;
    } else {
        firstNode = node;
    }
    lastNode = node;
    dfs(node -> right);
}

struct Node* treeToDoublyList(struct Node* root) {
    if (!root) return NULL;
    firstNode = NULL;
    lastNode = NULL;
    dfs(root);
    firstNode -> left = lastNode;
    lastNode -> right = firstNode;
    return firstNode;
}
