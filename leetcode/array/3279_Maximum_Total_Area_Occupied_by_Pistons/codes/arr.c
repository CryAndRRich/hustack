#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int key;
    int *vals;
    int size;
    int cap;
} Entry;

int cmp(const void *a, const void *b) {
    return ((Entry*)a) -> key - ((Entry*)b) -> key;
}

long long maxArea(int height, int* positions, int positionsSize, char* directions) {
    Entry *have = malloc(sizeof(Entry) * positionsSize * 2);
    int haveSize = 0;
    long long s = 0;
    int a = 0;
    for (int i = 0; i < positionsSize; i++) {
        s += positions[i];
        int k1, k2;
        if (directions[i] == 'U') {
            a++;
            k1 = height - positions[i];
            k2 = height - positions[i] + height;
        } else {
            a--;
            k1 = positions[i];
            k2 = positions[i] + height;
        }
        Entry *e1 = &have[haveSize++];
        e1 -> key = k1; e1 -> size = 0; e1 -> cap = 4; e1 -> vals = malloc(4 * sizeof(int)); e1 -> vals[e1 -> size++] = i;
        Entry *e2 = &have[haveSize++];
        e2 -> key = k2; e2 -> size = 0; e2 -> cap = 4; e2 -> vals = malloc(4 * sizeof(int)); e2 -> vals[e2 -> size++] = i;
    }
    qsort(have, haveSize, sizeof(Entry), cmp);
    int uniq = 0;
    for (int i = 0; i < haveSize; i++) {
        if (uniq > 0 && have[i].key == have[uniq-1].key) {
            Entry *dst = &have[uniq-1];
            for (int j = 0; j < have[i].size; j++) {
                if (dst -> size == dst -> cap) {
                    dst -> cap *= 2;
                    dst -> vals = realloc(dst -> vals, dst -> cap * sizeof(int));
                }
                dst -> vals[dst -> size++] = have[i].vals[j];
            }
            free(have[i].vals);
        } else {
            have[uniq++] = have[i];
        }
    }
    haveSize = uniq;
    long long r = s;
    int now = 0;
    for (int idx = 0; idx < haveSize; idx++) {
        long long t = have[idx].key - now;
        s += t * a;
        now = have[idx].key;
        if (s > r) r = s;
        for (int j = 0; j < have[idx].size; j++) {
            int i = have[idx].vals[j];
            if (directions[i] == 'U') {
                directions[i] = 'D';
                a -= 2;
            } else {
                directions[i] = 'U';
                a += 2;
            }
        }
    }
    for (int i = 0; i < haveSize; i++) free(have[i].vals);
    free(have);
    return r;
}
