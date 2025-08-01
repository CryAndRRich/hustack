#include <stdio.h>
#include <stdlib.h>

#define MAX 50001

typedef struct ListNode {
    int val;
    struct ListNode* next;
} ListNode;

ListNode* children[MAX];

void addChild(int parent, int child) {
    ListNode* node = (ListNode*)malloc(sizeof(ListNode));
    node -> val = child;
    node -> next = children[parent];
    children[parent] = node;
}

/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
int* killProcess(int* pid, int pidSize, int* ppid, int ppidSize, int kill, int* returnSize) {
    *returnSize = 0;
    memset(children, 0, sizeof(children));

    for (int i = 0; i < pidSize; i++) {
        addChild(ppid[i], pid[i]);
    }

    int* result = (int*)malloc(sizeof(int) * pidSize);
    int front = 0, back = 0;
    int* queue = (int*)malloc(sizeof(int) * pidSize);

    queue[back++] = kill;

    while (front < back) {
        int curr = queue[front++];
        result[*returnSize] = curr;
        (*returnSize)++;
        ListNode* child = children[curr];
        while (child) {
            queue[back++] = child -> val;
            child = child -> next;
        }
    }

    free(queue);
    return result;
}
