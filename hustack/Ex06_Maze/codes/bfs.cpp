#include <iostream>
#include <queue>
#include <vector>
using namespace std;

#define MAX 1000

int n, m, r, c;
int maze[MAX][MAX];
bool visited[MAX][MAX];

int main() {
    cin >> n >> m >> r >> c;
    r -= 1; 
    c -= 1;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> maze[i][j];
        }
    }
    
    if (maze[r][c] == 1) {
        cout << "-1";
        return 0;
    }
    
    if (r == 0 || r == n - 1 || c == 0 || c == m - 1) {
        cout << "1";
        return 0;
    }
    
    queue<vector<int>> q;
    q.push({r, c, 0});
    visited[r][c] = true;
    
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};
    
    while (!q.empty()) {
        vector<int> current = q.front();
        q.pop();
        
        int x = current[0], y = current[1], steps = current[2];
        if (x == 0 || x == n - 1 || y == 0 || y == m - 1) {
            cout << steps + 1;
            return 0;
        }
        
        for (int i = 0; i < 4; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m) {
                if (!visited[nx][ny] && maze[nx][ny] == 0) {
                    visited[nx][ny] = true;
                    q.push({nx, ny, steps + 1});
                }
            }
        }
    }
    
    cout << "-1";
    return 0;
}
