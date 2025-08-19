#include <stdlib.h>
#include <stdbool.h>

typedef struct Node {
    int val;
    struct Node* next;
} Node;

typedef struct {
    bool* isSlotAvailable;
    Node* head;
} PhoneDirectory;

PhoneDirectory* phoneDirectoryCreate(int maxNumbers) {
    PhoneDirectory* obj = (PhoneDirectory*)malloc(sizeof(PhoneDirectory));
    obj -> isSlotAvailable = (bool*)malloc(maxNumbers * sizeof(bool));
    for (int i = 0; i < maxNumbers; i++) obj -> isSlotAvailable[i] = true;
    obj -> head = (Node*)malloc(sizeof(Node));
    obj -> head -> val = -1;
    obj -> head -> next = NULL;

    Node* cur = obj -> head;
    for (int i = 0; i < maxNumbers; i++) {
        Node* newNode = (Node*)malloc(sizeof(Node));
        newNode -> val = i;
        newNode -> next = NULL;
        cur -> next = newNode;
        cur = newNode;
    }
    return obj;
}

int phoneDirectoryGet(PhoneDirectory* obj) {
    if (!obj -> head -> next) return -1;
    Node* node = obj -> head -> next;
    obj -> head -> next = node -> next;
    obj -> isSlotAvailable[node -> val] = false;
    int val = node -> val;
    free(node);
    return val;
}

bool phoneDirectoryCheck(PhoneDirectory* obj, int number) {
    return obj -> isSlotAvailable[number];
}

void phoneDirectoryRelease(PhoneDirectory* obj, int number) {
    if (obj -> isSlotAvailable[number]) return;
    Node* node = (Node*)malloc(sizeof(Node));
    node -> val = number;
    node -> next = obj -> head -> next;
    obj -> head -> next = node;
    obj -> isSlotAvailable[number] = true;
}

void phoneDirectoryFree(PhoneDirectory* obj) {
    Node* cur = obj -> head;
    while (cur) {
        Node* tmp = cur;
        cur = cur -> next;
        free(tmp);
    }
    free(obj -> isSlotAvailable);
    free(obj);
}

/**
 * Your PhoneDirectory struct will be instantiated and called as such:
 * PhoneDirectory* obj = phoneDirectoryCreate(maxNumbers);
 * int param_1 = phoneDirectoryGet(obj);
 
 * bool param_2 = phoneDirectoryCheck(obj, number);
 
 * phoneDirectoryRelease(obj, number);
 
 * phoneDirectoryFree(obj);
*/