#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int gcd(int x, int y) {
    while (y) {
        int temp = y;
        y = x % y;
        x = temp;
    }
    return x;
}

int bfs(int a, int b, int c) {
    if (c > (a > b ? a : b) || (c % gcd(a, b)) != 0) {
        return -1;
    }
    
    int capacity = (a + 1) * (b + 1);
    
    int *qx = (int *)malloc(capacity * sizeof(int));
    int *qy = (int *)malloc(capacity * sizeof(int));
    int *qsteps = (int *)malloc(capacity * sizeof(int));
    
    static char visited[1001][1001];
    memset(visited, 0, sizeof(visited));
    
    int head = 0, tail = 0;
    
    qx[tail] = 0; 
    qy[tail] = 0; 
    qsteps[tail] = 0;
    tail += 1;
    visited[0][0] = 1;
    
    while (head < tail) {
        int x = qx[head];
        int y = qy[head];
        int steps = qsteps[head];
        head += 1;
        
        if (x == c || y == c) {
            free(qx); free(qy); free(qsteps);
            return steps;
        }
        
        int nx, ny;
        
        nx = a; ny = y;
        if (!visited[nx][ny]) {
            visited[nx][ny] = 1;
            qx[tail] = nx; 
            qy[tail] = ny; 
            qsteps[tail] = steps + 1; 
            tail += 1;
        }
        nx = x; ny = b;
        if (!visited[nx][ny]) {
            visited[nx][ny] = 1;
            qx[tail] = nx; 
            qy[tail] = ny; 
            qsteps[tail] = steps + 1; 
            tail += 1;
        }
        nx = 0; ny = y;
        if (!visited[nx][ny]) {
            visited[nx][ny] = 1;
            qx[tail] = nx; 
            qy[tail] = ny; 
            qsteps[tail] = steps + 1; 
            tail += 1;
        }
        nx = x; ny = 0;
        if (!visited[nx][ny]) {
            visited[nx][ny] = 1;
            qx[tail] = nx; 
            qy[tail] = ny; 
            qsteps[tail] = steps + 1; 
            tail += 1;
        }
        int trans = fmin(x, b - y);
        nx = x - trans; ny = y + trans;
        if (!visited[nx][ny]) {
            visited[nx][ny] = 1;
            qx[tail] = nx; 
            qy[tail] = ny; 
            qsteps[tail] = steps + 1; 
            tail += 1;
        }
        trans = fmin(y, a - x);
        nx = x + trans; ny = y - trans;
        if (!visited[nx][ny]) {
            visited[nx][ny] = 1;
            qx[tail] = nx; 
            qy[tail] = ny; 
            qsteps[tail] = steps + 1; 
            tail += 1;
        }
    }
    
    free(qx); free(qy); free(qsteps);
    return -1;
}

int main() {
    int a, b, c;
    scanf("%d %d %d", &a, &b, &c);
    int result = bfs(a, b, c);
    printf("%d\n", result);
    return 0;
}
