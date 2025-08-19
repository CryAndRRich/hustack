#include <limits.h>
#include <stdlib.h>

struct Queue {
    int *data;
    int front, rear, size, capacity;
};

struct Queue* createQueue(int capacity) {
    struct Queue* q = (struct Queue*)malloc(sizeof(struct Queue));
    q -> capacity = capacity;
    q -> front = 0;
    q -> rear = -1;
    q -> size = 0;
    q -> data = (int*)malloc(sizeof(int) * capacity);
    return q;
}

int isEmpty(struct Queue* q) {
    return q -> size == 0;
}

void enqueue(struct Queue* q, int val) {
    q -> rear = (q -> rear + 1) % q -> capacity;
    q -> data[q -> rear] = val;
    q -> size++;
}

int dequeue(struct Queue* q) {
    int val = q -> data[q -> front];
    q -> front = (q -> front + 1) % q -> capacity;
    q -> size--;
    return val;
}

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* maxHammingDistances(int* nums, int numsSize, int m, int* returnSize) {
    int n = 1 << m;
    int *dist = (int*)malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) dist[i] = -1;
    struct Queue* q = createQueue(n);
    for (int i = 0; i < numsSize; i++) {
        dist[nums[i]] = 0;
        enqueue(q, nums[i]);
    }
    while (!isEmpty(q)) {
        int u = dequeue(q);
        for (int b = 0; b < m; b++) {
            int v = u ^ (1 << b);
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                enqueue(q, v);
            }
        }
    }
    int mask = n - 1;
    int* ans = (int*)malloc(sizeof(int) * numsSize);
    for (int i = 0; i < numsSize; i++) {
        ans[i] = m - dist[mask ^ nums[i]];
    }
    *returnSize = numsSize;
    free(q -> data);
    free(q);
    free(dist);
    return ans;
}
