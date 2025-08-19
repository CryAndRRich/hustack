#include <stdlib.h>

/**
Definition for a binary tree node.
 */
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

typedef struct {
    int key;
    int count;
} MapEntry;

static MapEntry map[10005];
static int mapSize;

static int getCount(int key) {
    for (int i = 0; i < mapSize; i++) {
        if (map[i].key == key) return map[i].count;
    }
    return 0;
}

static void addCount(int key) {
    for (int i = 0; i < mapSize; i++) {
        if (map[i].key == key) {
            map[i].count++;
            return;
        }
    }
    map[mapSize].key = key;
    map[mapSize].count = 1;
    mapSize++;
}

static void dfs(struct TreeNode* node) {
    if (!node) return;
    addCount(node -> val);
    if (getCount(node -> val) > 1) return;
    dfs(node -> left);
    dfs(node -> right);
}

static int maxDepth(struct TreeNode* node, int depth) {
    if (!node || getCount(node -> val) > 1) return depth;
    int left = maxDepth(node -> left, depth + 1);
    int right = maxDepth(node -> right, depth + 1);
    return left > right ? left : right;
}

int heightOfTree(struct TreeNode* root) {
    mapSize = 0;
    dfs(root);
    return maxDepth(root, 0);
}
