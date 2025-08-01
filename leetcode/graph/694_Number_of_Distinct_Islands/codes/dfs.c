#include <stdlib.h>
#include <string.h>

#define MAX_SHAPE 1000
#define MAX_LEN 100

typedef struct {
    int *points;
    int size;
} Shape;

int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};

void dfs(int** grid, int m, int n, int r0, int c0, int r, int c, int* shape, int* size) {
    if (r < 0 || r >= m || c < 0 || c >= n || grid[r][c] == 0)
        return;
    grid[r][c] = 0;
    shape[(*size)++] = (r - r0) * n + (c - c0);
    for (int i = 0; i < 4; ++i)
        dfs(grid, m, n, r0, c0, r + dirs[i][0], c + dirs[i][1], shape, size);
}

int compare(const void* a, const void* b) {
    return (*(int*)a - *(int*)b);
}

int numDistinctIslands(int** grid, int gridSize, int* gridColSize) {
    Shape shapes[MAX_SHAPE];
    int shapeCount = 0;

    for (int i = 0; i < gridSize; ++i) {
        for (int j = 0; j < gridColSize[0]; ++j) {
            if (grid[i][j] == 1) {
                int* shape = (int*)malloc(sizeof(int) * MAX_LEN);
                int size = 0;
                dfs(grid, gridSize, gridColSize[0], i, j, i, j, shape, &size);
                qsort(shape, size, sizeof(int), compare);

                int found = 0;
                for (int k = 0; k < shapeCount; ++k) {
                    if (shapes[k].size == size && memcmp(shapes[k].points, shape, sizeof(int) * size) == 0) {
                        found = 1;
                        break;
                    }
                }
                if (!found) {
                    shapes[shapeCount].points = shape;
                    shapes[shapeCount].size = size;
                    shapeCount++;
                } else {
                    free(shape);
                }
            }
        }
    }

    for (int i = 0; i < shapeCount; ++i)
        free(shapes[i].points);
    return shapeCount;
}
