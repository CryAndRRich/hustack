#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
int m, n, total, K;
int** G;
bool** visited;
int** path;
int pathSize;

bool backtrack(int x, int y, int count, int next) {
    if (visited[x][y]) return false;
    if (G[x][y] != 0 && G[x][y] != next) return false;
    visited[x][y] = true;
    path[pathSize][0] = x;
    path[pathSize][1] = y;
    pathSize++;
    int newNext = (G[x][y] == next ? next + 1 : next);
    if (count == total) {
        if (newNext == K + 1) return true;
        visited[x][y] = false;
        pathSize--;
        return false;
    }
    for (int d = 0; d < 4; d++) {
        int nx = x + dirs[d][0], ny = y + dirs[d][1];
        if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
            if (backtrack(nx, ny, count + 1, newNext)) return true;
        }
    }
    visited[x][y] = false;
    pathSize--;
    return false;
}

int** findPath(int** grid, int gridSize, int* gridColSize, int k, int* returnSize, int** returnColumnSizes) {
    m = gridSize;
    n = gridColSize[0];
    total = m * n;
    K = k;
    G = grid;
    visited = (bool**)malloc(m * sizeof(bool*));
    for (int i = 0; i < m; i++) {
        visited[i] = (bool*)calloc(n, sizeof(bool));
    }
    path = (int**)malloc(total * sizeof(int*));
    for (int i = 0; i < total; i++) path[i] = (int*)malloc(2 * sizeof(int));
    pathSize = 0;
    int** starts = (int**)malloc(m * n * sizeof(int*));
    int sc = 0;
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) if (G[i][j] == 0 || G[i][j] == 1) {
        starts[sc] = (int*)malloc(2 * sizeof(int));
        starts[sc][0] = i; starts[sc][1] = j;
        sc++;
    }
    for (int s = 0; s < sc; s++) {
        for (int i = 0; i < m; i++) memset(visited[i], 0, n * sizeof(bool));
        pathSize = 0;
        int sx = starts[s][0], sy = starts[s][1];
        if (backtrack(sx, sy, 1, 1)) {
            *returnSize = pathSize;
            *returnColumnSizes = (int*)malloc(pathSize * sizeof(int));
            for (int t = 0; t < pathSize; t++) (*returnColumnSizes)[t] = 2;
            for (int i = 0; i < sc; i++) free(starts[i]);
            free(starts);
            for (int i = 0; i < m; i++) free(visited[i]);
            free(visited);
            return path;
        }
    }
    for (int i = 0; i < sc; i++) free(starts[i]);
    free(starts);
    for (int i = 0; i < total; i++) free(path[i]);
    free(path);
    for (int i = 0; i < m; i++) free(visited[i]);
    free(visited);
    *returnSize = 0;
    *returnColumnSizes = NULL;
    return NULL;
}
