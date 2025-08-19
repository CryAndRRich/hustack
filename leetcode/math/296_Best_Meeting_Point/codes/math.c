#include <stdlib.h>

int cmp(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int minTotalDistance(int** grid, int gridSize, int* gridColSize) {
    int rows[40000], cols[40000]; 
    int rCount = 0, cCount = 0;

    for (int i = 0; i < gridSize; i++)
        for (int j = 0; j < gridColSize[0]; j++)
            if (grid[i][j] == 1)
                rows[rCount++] = i;

    for (int j = 0; j < gridColSize[0]; j++)
        for (int i = 0; i < gridSize; i++)
            if (grid[i][j] == 1)
                cols[cCount++] = j;

    qsort(rows, rCount, sizeof(int), cmp);
    qsort(cols, cCount, sizeof(int), cmp);

    int row_median = rows[rCount / 2];
    int col_median = cols[cCount / 2];
    int sum = 0;

    for (int i = 0; i < rCount; i++) sum += abs(rows[i] - row_median);
    for (int i = 0; i < cCount; i++) sum += abs(cols[i] - col_median);

    return sum;
}
