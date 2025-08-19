#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

typedef struct {
    int* table;       
    bool* used;      
    int size;         
    int count;      
} HashSet;

unsigned int hashFunction(int key, int size) {
    return (unsigned int)key % size;
}

HashSet* createHashSet(int size) {
    HashSet* set = (HashSet*)malloc(sizeof(HashSet));
    set -> size = size * 2; 
    set -> count = 0;
    set -> table = (int*)malloc(sizeof(int) * set -> size);
    set -> used = (bool*)malloc(sizeof(bool) * set -> size);
    for (int i = 0; i < set -> size; i++) set -> used[i] = false;
    return set;
}

bool hashSetContains(HashSet* set, int key) {
    unsigned int idx = hashFunction(key, set -> size);
    for (int i = 0; i < set -> size; i++) {
        unsigned int probe = (idx + i) % set -> size;
        if (!set -> used[probe]) return false;
        if (set -> table[probe] == key) return true;
    }
    return false;
}

void hashSetInsert(HashSet* set, int key) {
    if (hashSetContains(set, key)) return;
    unsigned int idx = hashFunction(key, set -> size);
    while (set -> used[idx]) {
        idx = (idx + 1) % set -> size;
    }
    set -> table[idx] = key;
    set -> used[idx] = true;
    set -> count++;
}

bool hashSetRemove(HashSet* set, int key) {
    unsigned int idx = hashFunction(key, set -> size);
    for (int i = 0; i < set -> size; i++) {
        unsigned int probe = (idx + i) % set -> size;
        if (!set -> used[probe]) return false;
        if (set -> table[probe] == key) {
            set -> used[probe] = false;
            set -> count--;
            return true;
        }
    }
    return false;
}

void freeHashSet(HashSet* set) {
    free(set -> table);
    free(set -> used);
    free(set);
}

typedef struct {
    HashSet* slotsAvailable;
} PhoneDirectory;

PhoneDirectory* phoneDirectoryCreate(int maxNumbers) {
    PhoneDirectory* obj = (PhoneDirectory*)malloc(sizeof(PhoneDirectory));
    obj -> slotsAvailable = createHashSet(maxNumbers);
    for (int i = 0; i < maxNumbers; i++) {
        hashSetInsert(obj -> slotsAvailable, i);
    }
    return obj;
}

int phoneDirectoryGet(PhoneDirectory* obj) {
    if (obj -> slotsAvailable -> count == 0) return -1;
    for (int i = 0; i < obj -> slotsAvailable -> size; i++) {
        if (obj -> slotsAvailable -> used[i]) {
            int val = obj -> slotsAvailable -> table[i];
            hashSetRemove(obj -> slotsAvailable, val);
            return val;
        }
    }
    return -1;
}

bool phoneDirectoryCheck(PhoneDirectory* obj, int number) {
    return hashSetContains(obj -> slotsAvailable, number);
}

void phoneDirectoryRelease(PhoneDirectory* obj, int number) {
    hashSetInsert(obj -> slotsAvailable, number);
}

void phoneDirectoryFree(PhoneDirectory* obj) {
    freeHashSet(obj -> slotsAvailable);
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