#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    bool* isSlotAvailable;
    int* queue;
    int front;
    int rear;
    int size;
    int capacity;
} PhoneDirectory;

PhoneDirectory* phoneDirectoryCreate(int maxNumbers) {
    PhoneDirectory* obj = (PhoneDirectory*)malloc(sizeof(PhoneDirectory));
    obj -> isSlotAvailable = (bool*)malloc(maxNumbers * sizeof(bool));
    obj -> queue = (int*)malloc(maxNumbers * sizeof(int));
    obj -> front = 0;
    obj -> rear = 0;
    obj -> size = maxNumbers;
    obj -> capacity = maxNumbers;
    for (int i = 0; i < maxNumbers; i++) {
        obj -> isSlotAvailable[i] = true;
        obj -> queue[i] = i;
    }
    obj -> rear = maxNumbers;
    return obj;
}

int phoneDirectoryGet(PhoneDirectory* obj) {
    if (obj -> front == obj -> rear) {
        return -1;
    }
    int slot = obj -> queue[obj -> front++];
    if (obj -> front == obj -> capacity) obj -> front = 0; 
    obj -> isSlotAvailable[slot] = false;
    return slot;
}

bool phoneDirectoryCheck(PhoneDirectory* obj, int number) {
    return obj -> isSlotAvailable[number];
}

void phoneDirectoryRelease(PhoneDirectory* obj, int number) {
    if (obj -> isSlotAvailable[number]) {
        return;
    }
    obj -> queue[obj -> rear++] = number;
    if (obj -> rear == obj -> capacity) obj -> rear = 0; 
    obj -> isSlotAvailable[number] = true;
}

void phoneDirectoryFree(PhoneDirectory* obj) {
    free(obj -> isSlotAvailable);
    free(obj -> queue);
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