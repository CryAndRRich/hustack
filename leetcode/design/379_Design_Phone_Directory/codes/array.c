#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    bool* isSlotAvailable;
    int size;
} PhoneDirectory;

PhoneDirectory* phoneDirectoryCreate(int maxNumbers) {
    PhoneDirectory* obj = (PhoneDirectory*)malloc(sizeof(PhoneDirectory));
    obj -> isSlotAvailable = (bool*)malloc(maxNumbers * sizeof(bool));
    obj -> size = maxNumbers;
    for (int i = 0; i < maxNumbers; i++) {
        obj -> isSlotAvailable[i] = true;
    }
    return obj;
}

int phoneDirectoryGet(PhoneDirectory* obj) {
    for (int i = 0; i < obj -> size; i++) {
        if (obj -> isSlotAvailable[i]) {
            obj -> isSlotAvailable[i] = false;
            return i;
        }
    }
    return -1;
}

bool phoneDirectoryCheck(PhoneDirectory* obj, int number) {
    return obj -> isSlotAvailable[number];
}

void phoneDirectoryRelease(PhoneDirectory* obj, int number) {
    obj -> isSlotAvailable[number] = true;
}

void phoneDirectoryFree(PhoneDirectory* obj) {
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