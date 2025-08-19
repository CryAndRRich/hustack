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
    int heap[15];
    int size;
} Heap;

void heapPush(Heap* h, int val) {
    h -> heap[h -> size++] = val;
    int i = h -> size - 1;
    while (i > 0 && h -> heap[(i - 1) / 2] < h -> heap[i]) {
        int tmp = h -> heap[i];
        h -> heap[i] = h -> heap[(i - 1) / 2];
        h -> heap[(i - 1) / 2] = tmp;
        i = (i - 1) / 2;
    }
}

void heapPop(Heap* h) {
    if (h -> size == 0) return;
    h -> heap[0] = h -> heap[--h -> size];
    int i = 0;
    while (1) {
        int l = 2i + 1, r = 2i + 2, largest = i;
        if (l < h -> size && h -> heap[l] > h -> heap[largest]) largest = l;
        if (r < h -> size && h -> heap[r] > h -> heap[largest]) largest = r;
        if (largest == i) break;
        int tmp = h -> heap[i];
        h -> heap[i] = h -> heap[largest];
        h -> heap[largest] = tmp;
        i = largest;
    }
}

typedef struct {
    Heap heap;
    int count;
} Result;

Result dfs(struct TreeNode* root, int k) {
    Result res;
    res.heap.size = 0;
    res.count = 0;
    if (!root) return res;

    Result left = dfs(root -> left, k);
    Result right = dfs(root -> right, k);

    res.heap = left.heap;
    for (int i = 0; i < right.heap.size; i++) {
        heapPush(&res.heap, right.heap.heap[i]);
        if (res.heap.size > k) heapPop(&res.heap);
    }

    res.count = left.count + right.count;
    if (res.heap.size == k && res.heap.heap[0] < root -> val) res.count++;

    heapPush(&res.heap, root -> val);
    if (res.heap.size > k) heapPop(&res.heap);

    return res;
}

int countGreatEnoughNodes(struct TreeNode* root, int k) {
    return dfs(root, k).count;
}
