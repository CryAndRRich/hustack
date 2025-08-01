#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define MAX_WORKERS 1000
#define MAX_BIKES 1000

typedef struct {
    int dist;
    int worker;
    int bike;
} Triple;

int cmp(const void* a, const void* b) {
    Triple* pa = (Triple*)a;
    Triple* pb = (Triple*)b;
    if (pa -> dist != pb -> dist) return pa -> dist - pb -> dist;
    if (pa -> worker != pb -> worker) return pa -> worker - pb -> worker;
    return pa -> bike - pb -> bike;
}

int* assignBikes(int** workers, int workersSize, int* workersColSize,
                 int** bikes, int bikesSize, int* bikesColSize, int* returnSize) {
    bool assignedWorker[MAX_WORKERS] = { false };
    bool usedBike[MAX_BIKES] = { false };
    int* result = (int*)malloc(sizeof(int) * workersSize);
    for (int i = 0; i < workersSize; ++i) result[i] = -1;

    int deviation = 1;
    int remaining = workersSize;

    while (remaining > 0) {
        Triple pool[MAX_WORKERS * MAX_BIKES];
        int poolSize = 0;

        for (int i = 0; i < workersSize; ++i) {
            if (assignedWorker[i]) continue;
            for (int j = 0; j < bikesSize; ++j) {
                if (usedBike[j]) continue;

                int dist = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1]);
                if (dist <= deviation) {
                    pool[poolSize++] = (Triple){dist, i, j};
                }
            }
        }

        qsort(pool, poolSize, sizeof(Triple), cmp);

        for (int i = 0; i < poolSize; ++i) {
            int w = pool[i].worker;
            int b = pool[i].bike;
            if (!assignedWorker[w] && !usedBike[b]) {
                result[w] = b;
                assignedWorker[w] = true;
                usedBike[b] = true;
                remaining--;
            }
        }

        deviation <<= 1;
    }

    *returnSize = workersSize;
    return result;
}
