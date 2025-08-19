#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int timestamp;
    int count;
    struct Node* next;
} Node;

typedef struct {
    int total;
    Node* front;
    Node* rear;
} HitCounter;

HitCounter* hitCounterCreate() {
    HitCounter* obj = (HitCounter*)malloc(sizeof(HitCounter));
    obj -> total = 0;
    obj -> front = obj -> rear = NULL;
    return obj;
}

void hitCounterHit(HitCounter* obj, int timestamp) {
    if (obj -> rear && obj -> rear -> timestamp == timestamp) {
        obj -> rear -> count++;
    } else {
        Node* node = (Node*)malloc(sizeof(Node));
        node -> timestamp = timestamp;
        node -> count = 1;
        node -> next = NULL;
        if (obj -> rear) obj -> rear -> next = node;
        else obj -> front = node;
        obj -> rear = node;
    }
    obj -> total++;
}

int hitCounterGetHits(HitCounter* obj, int timestamp) {
    while (obj -> front && timestamp - obj -> front -> timestamp >= 300) {
        obj -> total -= obj -> front -> count;
        Node* temp = obj -> front;
        obj -> front = obj -> front -> next;
        free(temp);
        if (!obj -> front) obj -> rear = NULL;
    }
    return obj -> total;
}

void hitCounterFree(HitCounter* obj) {
    Node* cur = obj -> front;
    while (cur) {
        Node* tmp = cur;
        cur = cur -> next;
        free(tmp);
    }
    free(obj);
}
