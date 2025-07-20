#include <stdio.h>
#include <stdlib.h>

#define MAX 1000
#define INF 1000000000

typedef struct {
    int x, y, steps;
} Node;

int n, m, r, c;
int maze[MAX][MAX];
int visited[MAX][MAX];

int dx[4] = {-1, 1, 0, 0};
int dy[4] = {0, 0, -1, 1};

int minSteps = INF;

void dfs(int x, int y, int steps) {
    if (steps >= minSteps) return;
    if (x == 0 || x == n - 1 || y == 0 || y == m - 1) {
        if (steps + 1 < minSteps)
            minSteps = steps + 1;
        return;
    }
    
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx >= 0 && nx < n && ny >= 0 && ny < m && !visited[nx][ny] && maze[nx][ny] == 0) {
            visited[nx][ny] = 1;
            dfs(nx, ny, steps + 1);
            visited[nx][ny] = 0;
        }
    }
}

int main() {
    scanf("%d %d %d %d", &n, &m, &r, &c);
    r -= 1; 
    c -= 1;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            scanf("%d", &maze[i][j]);
        }
    }
    
    if (maze[r][c] == 1) {
        printf("-1");
        return 0;
    }
    
    if (r == 0 || r == n - 1 || c == 0 || c == m - 1) {
        printf("1");
        return 0;
    }
    
    visited[r][c] = 1;
    dfs(r, c, 0);
    
    if (minSteps == INF)
        printf("-1");
    else
        printf("%d", minSteps);
        
    return 0;
}
