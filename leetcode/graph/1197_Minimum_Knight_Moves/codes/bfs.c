#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#define MAX 605

typedef struct {
    int x, y, steps;
} State;

bool visited[MAX][MAX];

int minKnightMoves(int x, int y) {
    int dirs[8][2] = {{2,1},{1,2},{-1,2},{-2,1},{-2,-1},{-1,-2},{1,-2},{2,-1}};
    x = abs(x); y = abs(y);

    memset(visited, 0, sizeof(visited));
    State queue[MAX * MAX];
    int front = 0, rear = 0;

    int offset = 302; 
    visited[offset][offset] = true;
    queue[rear++] = (State){0, 0, 0};

    while (front < rear) {
        State curr = queue[front++];
        if (curr.x == x && curr.y == y) return curr.steps;
        for (int i = 0; i < 8; ++i) {
            int nx = curr.x + dirs[i][0];
            int ny = curr.y + dirs[i][1];
            if (nx >= -1 && ny >= -1 && nx <= x + 2 && ny <= y + 2) {
                int xi = nx + offset, yi = ny + offset;
                if (!visited[xi][yi]) {
                    visited[xi][yi] = true;
                    queue[rear++] = (State){nx, ny, curr.steps + 1};
                }
            }
        }
    }
    return -1;
}
