#include <stdio.h>
#include <stdlib.h>

#define MAX 1000

typedef struct {
    int x, y, steps;
} Node;

int n, m, r, c;
int maze[MAX][MAX];
int visited[MAX][MAX];

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
    int queueSize = n * m;
    Node *queue = (Node*) malloc(queueSize * sizeof(Node));
    int start = 0, end = 0;
    
    Node first = {r, c, 0};
    queue[end++] = first;
    visited[r][c] = 1;
    
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};
    
    while (start < end) {
        Node current = queue[start++];
        int x = current.x, y = current.y, steps = current.steps;
        if (x == 0 || x == n - 1 || y == 0 || y == m - 1) {
            printf("%d", steps + 1);
            free(queue);
            return 0;
        }
        
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m) {
                if (!visited[nx][ny] && maze[nx][ny] == 0) {
                    visited[nx][ny] = 1;
                    Node next = {nx, ny, steps + 1};
                    queue[end++] = next;
                }
            }
        }
    }
    
    printf("-1");
    free(queue);
    return 0;
}
