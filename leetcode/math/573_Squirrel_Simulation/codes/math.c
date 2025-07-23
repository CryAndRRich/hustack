#include <stdlib.h>
#include <limits.h>

int absDiff(int a, int b) {
    return (a > b) ? (a - b) : (b - a);
}

int minDistance(int height, int width, int* tree, int treeSize, int* squirrel, int squirrelSize, int** nuts, int nutsSize, int* nutsColSize) {
    int total = 0;
    int maxSaving = INT_MIN;

    for (int i = 0; i < nutsSize; ++i) {
        int* nut = nuts[i];
        int distTree = absDiff(nut[0], tree[0]) + absDiff(nut[1], tree[1]);
        int distSquirrel = absDiff(nut[0], squirrel[0]) + absDiff(nut[1], squirrel[1]);
        total += 2 * distTree;
        int saving = distTree - distSquirrel;
        if (saving > maxSaving) {
            maxSaving = saving;
        }
    }

    return total - maxSaving;
}
