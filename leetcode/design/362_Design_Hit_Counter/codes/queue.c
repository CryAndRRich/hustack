#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int timestamp;
    struct Node* next;
} Node;

typedef struct {
    Node* front;
    Node* rear;
} HitCounter;

HitCounter* hitCounterCreate() {
    HitCounter* obj = (HitCounter*)malloc(sizeof(HitCounter));
    obj -> front = obj -> rear = NULL;
    return obj;
}

void hitCounterHit(HitCounter* obj, int timestamp) {
    Node* node = (Node*)malloc(sizeof(Node));
    node -> timestamp = timestamp;
    node -> next = NULL;
    if (obj -> rear) obj -> rear -> next = node;
    else obj -> front = node;
    obj -> rear = node;
}

int hitCounterGetHits(HitCounter* obj, int timestamp) {
    while (obj -> front && timestamp - obj -> front -> timestamp >= 300) {
        Node* temp = obj -> front;
        obj -> front = obj -> front -> next;
        free(temp);
        if (!obj -> front) obj -> rear = NULL;
    }
    int count = 0;
    Node* cur = obj -> front;
    while (cur) {
        count++;
        cur = cur -> next;
    }
    return count;
}

void hitCounterFree(HitCounter* obj) {
    Node* cur = obj -> front;
    while (cur) {
        Node* temp = cur;
        cur = cur -> next;
        free(temp);
    }
    free(obj);
}
