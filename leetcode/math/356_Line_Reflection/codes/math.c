#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#define HASH_SIZE 100003

typedef struct Node {
    char* key;
    struct Node* next;
} Node;

Node* hashTable[HASH_SIZE];

unsigned int hash(const char* str) {
    unsigned int h = 0;
    while (*str)
        h = (h * 131 + *str++) % HASH_SIZE;
    return h;
}

bool find(const char* key) {
    unsigned int idx = hash(key);
    Node* node = hashTable[idx];
    while (node) {
        if (strcmp(node->key, key) == 0)
            return true;
        node = node->next;
    }
    return false;
}

void insert(const char* key) {
    unsigned int idx = hash(key);
    Node* node = malloc(sizeof(Node));
    node->key = strdup(key);
    node->next = hashTable[idx];
    hashTable[idx] = node;
}

void clearHashTable() {
    for (int i = 0; i < HASH_SIZE; i++) {
        Node* node = hashTable[i];
        while (node) {
            Node* next = node->next;
            free(node->key);
            free(node);
            node = next;
        }
        hashTable[i] = NULL;
    }
}

bool isReflected(int** points, int pointsSize, int* pointsColSize) {
    int minX = INT_MAX, maxX = INT_MIN;
    char buf[64];

    for (int i = 0; i < pointsSize; ++i) {
        int x = points[i][0], y = points[i][1];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        snprintf(buf, sizeof(buf), "%d#%d", x, y);
        insert(buf);
    }

    int reflectX = minX + maxX;

    for (int i = 0; i < pointsSize; ++i) {
        int x = points[i][0], y = points[i][1];
        int reflectedX = reflectX - x;
        snprintf(buf, sizeof(buf), "%d#%d", reflectedX, y);
        if (!find(buf)) {
            clearHashTable();
            return false;
        }
    }

    clearHashTable();
    return true;
}
