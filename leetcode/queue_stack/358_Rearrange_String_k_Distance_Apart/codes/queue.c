#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define MAX 26
#define MAX_LEN 300005

typedef struct {
    int count;
    char ch;
    int time;
} Node;

typedef struct {
    Node data[MAX_LEN];
    int front, rear;
} Queue;

void initQueue(Queue* q) {
    q -> front = q -> rear = 0;
}

bool isEmpty(Queue* q) {
    return q -> front == q -> rear;
}

void enqueue(Queue* q, Node val) {
    q -> data[q -> rear++] = val;
}

Node dequeue(Queue* q) {
    return q -> data[q -> front++];
}

char* rearrangeString(char* s, int k) {
    if (k == 0) return strdup(s);

    int freq[26] = {0};
    for (int i = 0; s[i]; ++i) freq[s[i] - 'a']++;

    Node heap[MAX];
    int heapSize = 0;

    for (int i = 0; i < 26; ++i)
        if (freq[i] > 0)
            heap[heapSize++] = (Node){.count = freq[i], .ch = 'a' + i};

    Queue waitQ;
    initQueue(&waitQ);

    char* res = malloc(strlen(s) + 1);
    int len = 0;

    while (heapSize > 0 || !isEmpty(&waitQ)) {
        if (!isEmpty(&waitQ) && waitQ.data[waitQ.front].time <= len) {
            heap[heapSize++] = dequeue(&waitQ);
        }

        int maxIndex = -1, maxCount = -1;
        for (int i = 0; i < heapSize; ++i)
            if (heap[i].count > maxCount) {
                maxCount = heap[i].count;
                maxIndex = i;
            }

        if (maxIndex == -1) {
            free(res);
            return strdup("");
        }

        Node chosen = heap[maxIndex];
        res[len++] = chosen.ch;

        for (int i = maxIndex; i < heapSize - 1; ++i)
            heap[i] = heap[i + 1];
        heapSize--;

        if (chosen.count - 1 > 0) {
            enqueue(&waitQ, (Node){.count = chosen.count - 1, .ch = chosen.ch, .time = len + k - 1});
        }
    }

    res[len] = '\0';
    return res;
}
