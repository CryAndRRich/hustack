/**
 * Note: The returned array must be malloced, assume caller calls free().
 */
#include <stdlib.h>
#include <string.h>

typedef struct {
    int key;
    int val;
} Entry;

#define MAX 10000

int hash(int key) {
    return key % MAX;
}

int* anagramMappings(int* nums1, int nums1Size, int* nums2, int nums2Size, int* returnSize) {
    Entry* map[MAX] = {NULL};
    int* result = (int*)malloc(sizeof(int) * nums1Size);
    *returnSize = nums1Size;

    for (int i = 0; i < nums2Size; i++) {
        int h = hash(nums2[i]);
        while (map[h] != NULL && map[h] -> key != nums2[i])
            h = (h + 1) % MAX;
        if (!map[h]) {
            map[h] = (Entry*)malloc(sizeof(Entry));
            map[h] -> key = nums2[i];
            map[h] -> val = i;
        }
    }

    for (int i = 0; i < nums1Size; i++) {
        int h = hash(nums1[i]);
        while (map[h] -> key != nums1[i])
            h = (h + 1) % MAX;
        result[i] = map[h] -> val;
    }

    return result;
}
