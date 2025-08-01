#include <stdlib.h>
#include <stdbool.h>

struct ZigzagIterator {
    int* v1;
    int v1Size;
    int* v2;
    int v2Size;
    int i1;
    int i2;
    int turn;
};

struct ZigzagIterator *zigzagIteratorCreate(int* v1, int v1Size, int* v2, int v2Size) {
    struct ZigzagIterator* it = malloc(sizeof(struct ZigzagIterator));
    it -> v1 = v1;
    it -> v1Size = v1Size;
    it -> v2 = v2;
    it -> v2Size = v2Size;
    it -> i1 = 0;
    it -> i2 = 0;
    it -> turn = 0;
    return it;
}

bool zigzagIteratorHasNext(struct ZigzagIterator *iter) {
    return iter -> i1 < iter -> v1Size || iter -> i2 < iter -> v2Size;
}

int zigzagIteratorNext(struct ZigzagIterator *iter) {
    for (int _ = 0; _ < 2; _++) {
        if (iter -> turn % 2 == 0 && iter -> i1 < iter -> v1Size) {
            iter -> turn++;
            return iter -> v1[iter -> i1++];
        } else if (iter -> turn % 2 == 1 && iter -> i2 < iter -> v2Size) {
            iter -> turn++;
            return iter -> v2[iter -> i2++];
        }
        iter -> turn++;
    }
    return -1;
}

void zigzagIteratorFree(struct ZigzagIterator *iter) {
    free(iter);
}

/**
 * Your ZigzagIterator will be called like this:
 * struct ZigzagIterator *i = zigzagIteratorCreate(v1, v1Size, v2, v2Size);
 * while (zigzagIteratorHasNext(i)) printf("%d\n", zigzagIteratorNext(i));
 * zigzagIteratorFree(i);
 */