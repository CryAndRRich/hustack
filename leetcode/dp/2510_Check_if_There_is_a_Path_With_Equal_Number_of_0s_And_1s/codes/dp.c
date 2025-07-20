#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

bool dfs(int** grid, int i, int j, int diff, int offset, int m, int n, char*** memo) {
    if (i < 0 || j < 0) return false;

    diff += (grid[i][j] == 0 ? 1 : -1);
    if (diff < -offset || diff > offset) return false;

    if (memo[i][j][diff + offset] != -1)
        return memo[i][j][diff + offset];

    if (i == 0 && j == 0)
        return (memo[i][j][diff + offset] = (diff == 0));

    bool res = false;
    if (i > 0) res |= dfs(grid, i - 1, j, diff, offset, m, n, memo);
    if (j > 0) res |= dfs(grid, i, j - 1, diff, offset, m, n, memo);

    return memo[i][j][diff + offset] = res;
}

bool isThereAPath(int** grid, int gridSize, int* gridColSize) {
    int m = gridSize, n = gridColSize[0];
    int maxDiff = m + n;

    char*** memo = malloc(m * sizeof(char**));
    for (int i = 0; i < m; ++i) {
        memo[i] = malloc(n * sizeof(char*));
        for (int j = 0; j < n; ++j) {
            memo[i][j] = malloc((2 * maxDiff + 1) * sizeof(char));
            memset(memo[i][j], -1, (2 * maxDiff + 1));
        }
    }

    bool result = dfs(grid, m - 1, n - 1, 0, maxDiff, m, n, memo);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) free(memo[i][j]);
        free(memo[i]);
    }
    free(memo);

    return result;
}
