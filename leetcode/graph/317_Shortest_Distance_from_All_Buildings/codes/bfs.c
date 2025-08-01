#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>

#define MAX 50

int dir[4][2] = {{0,1}, {1,0}, {0,-1}, {-1,0}};

void bfs(int** grid, int m, int n, int** dist, int** reach, int sx, int sy) {
    bool visited[MAX][MAX] = {false};
    int q[MAX*MAX][2], front = 0, back = 0;
    q[back][0] = sx; q[back++][1] = sy;
    int level = 0;

    while (front < back) {
        int size = back - front;
        level++;
        for (int i = 0; i < size; ++i) {
            int x = q[front][0], y = q[front++][1];
            for (int d = 0; d < 4; ++d) {
                int nx = x + dir[d][0], ny = y + dir[d][1];
                if (nx >= 0 && ny >= 0 && nx < m && ny < n &&
                    !visited[nx][ny] && grid[nx][ny] == 0) {
                    visited[nx][ny] = true;
                    dist[nx][ny] += level;
                    reach[nx][ny]++;
                    q[back][0] = nx; q[back++][1] = ny;
                }
            }
        }
    }
}

int shortestDistance(int** grid, int gridSize, int* gridColSize){
    int m = gridSize, n = gridColSize[0];
    int** dist = malloc(sizeof(int*) * m);
    int** reach = malloc(sizeof(int*) * m);
    for (int i = 0; i < m; i++) {
        dist[i] = calloc(n, sizeof(int));
        reach[i] = calloc(n, sizeof(int));
    }

    int buildings = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (grid[i][j] == 1) {
                buildings++;
                bfs(grid, m, n, dist, reach, i, j);
            }

    int minDist = INT_MAX;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (grid[i][j] == 0 && reach[i][j] == buildings)
                if (dist[i][j] < minDist)
                    minDist = dist[i][j];

    for (int i = 0; i < m; i++) {
        free(dist[i]);
        free(reach[i]);
    }
    free(dist);
    free(reach);
    
    return minDist == INT_MAX ? -1 : minDist;
}
