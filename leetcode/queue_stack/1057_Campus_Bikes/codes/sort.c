#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

typedef struct {
    int dist;
    int worker;
    int bike;
} Pair;

int compare(const void* a, const void* b) {
    Pair* pa = (Pair*)a;
    Pair* pb = (Pair*)b;
    if (pa -> dist != pb -> dist) return pa -> dist - pb -> dist;
    if (pa -> worker != pb -> worker) return pa -> worker - pb -> worker;
    return pa -> bike - pb -> bike;
}

int* assignBikes(int** workers, int workersSize, int* workersColSize,
                 int** bikes, int bikesSize, int* bikesColSize, int* returnSize) {
    int totalPairs = workersSize * bikesSize;
    Pair* pairs = (Pair*)malloc(sizeof(Pair) * totalPairs);
    int idx = 0;

    for (int i = 0; i < workersSize; ++i) {
        for (int j = 0; j < bikesSize; ++j) {
            int dist = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1]);
            pairs[idx++] = (Pair){dist, i, j};
        }
    }

    qsort(pairs, totalPairs, sizeof(Pair), compare);

    int* result = (int*)malloc(sizeof(int) * workersSize);
    bool* assigned = (bool*)calloc(workersSize, sizeof(bool));
    bool* used = (bool*)calloc(bikesSize, sizeof(bool));
    for (int i = 0; i < workersSize; ++i) result[i] = -1;

    for (int i = 0; i < totalPairs; ++i) {
        int w = pairs[i].worker, b = pairs[i].bike;
        if (!assigned[w] && !used[b]) {
            result[w] = b;
            assigned[w] = true;
            used[b] = true;
        }
    }

    *returnSize = workersSize;
    free(pairs);
    free(assigned);
    free(used);
    return result;
}
