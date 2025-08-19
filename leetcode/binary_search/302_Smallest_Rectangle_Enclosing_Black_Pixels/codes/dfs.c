#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>

void dfs(char** image, int m, int n, int x, int y,
         bool** visited, int* min_x, int* max_x, int* min_y, int* max_y) {
    if (x < 0 || x >= m || y < 0 || y >= n) return;
    if (image[x][y] != '1' || visited[x][y]) return;

    visited[x][y] = true;
    if (x < *min_x) *min_x = x;
    if (x > *max_x) *max_x = x;
    if (y < *min_y) *min_y = y;
    if (y > *max_y) *max_y = y;

    dfs(image, m, n, x + 1, y, visited, min_x, max_x, min_y, max_y);
    dfs(image, m, n, x - 1, y, visited, min_x, max_x, min_y, max_y);
    dfs(image, m, n, x, y + 1, visited, min_x, max_x, min_y, max_y);
    dfs(image, m, n, x, y - 1, visited, min_x, max_x, min_y, max_y);
}

int minArea(char** image, int imageSize, int* imageColSize, int x, int y) {
    int m = imageSize, n = imageColSize[0];

    bool** visited = (bool**)malloc(m * sizeof(bool*));
    for (int i = 0; i < m; i++) {
        visited[i] = (bool*)calloc(n, sizeof(bool));
    }

    int min_x = INT_MAX, max_x = INT_MIN;
    int min_y = INT_MAX, max_y = INT_MIN;

    dfs(image, m, n, x, y, visited, &min_x, &max_x, &min_y, &max_y);

    for (int i = 0; i < m; i++) free(visited[i]);
    free(visited);

    return (max_x - min_x + 1) * (max_y - min_y + 1);
}
