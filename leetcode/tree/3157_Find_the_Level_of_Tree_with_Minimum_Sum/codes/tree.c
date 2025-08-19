/**
Definition for a binary tree node.
 */
#include <limits.h>
#include <stdlib.h>

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
};

struct QueueNode {
    struct TreeNode* node;
    struct QueueNode* next;
};

struct Queue {
    struct QueueNode* front;
    struct QueueNode* rear;
};

void enqueue(struct Queue* q, struct TreeNode* node) {
    struct QueueNode* newNode = (struct QueueNode*)malloc(sizeof(struct QueueNode));
    newNode -> node = node;
    newNode -> next = NULL;
    if (!q -> rear) {
        q -> front = q -> rear = newNode;
    } else {
        q -> rear -> next = newNode;
        q -> rear = newNode;
    }
}

struct TreeNode* dequeue(struct Queue* q) {
    if (!q -> front) return NULL;
    struct QueueNode* temp = q -> front;
    struct TreeNode* node = temp -> node;
    q -> front = q -> front -> next;
    if (!q -> front) q -> rear = NULL;
    free(temp);
    return node;
}

int queueSize(struct Queue* q) {
    int count = 0;
    struct QueueNode* cur = q -> front;
    while (cur) {
        count++;
        cur = cur -> next;
    }
    return count;
}

int minimumLevel(struct TreeNode* root) {
    struct Queue q = {NULL, NULL};
    enqueue(&q, root);
    int level = 1, ans = 1;
    long long minSum = LLONG_MAX;
    while (q.front) {
        int size = queueSize(&q);
        long long sum = 0;
        for (int i = 0; i < size; i++) {
            struct TreeNode* node = dequeue(&q);
            sum += node -> val;
            if (node -> left) enqueue(&q, node -> left);
            if (node -> right) enqueue(&q, node -> right);
        }
        if (sum < minSum) {
            minSum = sum;
            ans = level;
        }
        level++;
    }
    return ans;
}
