#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define MAX 100
#define MAX_PATH 300

typedef struct {
    int x, y, dist;
    char path[MAX_PATH];
} State;

const int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
const char* dirChars = "durl";

int inBounds(int x, int y, int m, int n, int** maze) {
    return x >= 0 && x < m && y >= 0 && y < n && maze[x][y] == 0;
}

char* findShortestWay(int** maze, int mazeSize, int* mazeColSize, int* ball, int ballSize, int* hole, int holeSize) {
    int m = mazeSize, n = mazeColSize[0];
    int dist[MAX][MAX];
    char path[MAX][MAX][MAX_PATH];

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            dist[i][j] = INT_MAX;
            path[i][j][0] = '\0';
        }

    State queue[10000];
    int front = 0, back = 0;

    dist[ball[0]][ball[1]] = 0;
    queue[back++] = (State){ball[0], ball[1], 0, ""};

    while (front < back) {
        State cur = queue[front++];

        for (int d = 0; d < 4; ++d) {
            int x = cur.x, y = cur.y, steps = 0;

            while (inBounds(x + dirs[d][0], y + dirs[d][1], m, n, maze)) {
                x += dirs[d][0];
                y += dirs[d][1];
                steps++;
                if (x == hole[0] && y == hole[1]) break;
            }

            int totalDist = cur.dist + steps;
            char newPath[MAX_PATH];
            snprintf(newPath, sizeof(newPath), "%s%c", cur.path, dirChars[d]);

            if (totalDist < dist[x][y] ||
                (totalDist == dist[x][y] && strcmp(newPath, path[x][y]) < 0)) {
                dist[x][y] = totalDist;
                strcpy(path[x][y], newPath);
                queue[back++] = (State){x, y, totalDist, ""};
                strcpy(queue[back - 1].path, newPath);
            }
        }
    }

    if (dist[hole[0]][hole[1]] == INT_MAX)
        return strdup("impossible");
    return strdup(path[hole[0]][hole[1]]);
}
