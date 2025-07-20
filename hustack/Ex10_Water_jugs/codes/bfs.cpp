#include <iostream>
#include <queue>
#include <tuple>
#include <set>
#include <algorithm>
#include <cstdlib>
#include <cstring>
using namespace std;

int gcd(int x, int y) {
    while (y) {
        int temp = y;
        y = x % y;
        x = temp;
    }
    return x;
}

int bfs(int a, int b, int c) {
    if (c > max(a, b) || c % gcd(a, b) != 0) {
        return -1;
    }
    
    bool visited[1001][1001];
    memset(visited, false, sizeof(visited));
    
    queue<tuple<int, int, int>> q;
    q.push(make_tuple(0, 0, 0));
    visited[0][0] = true;
    
    while (!q.empty()) {
        auto [x, y, steps] = q.front();
        q.pop();
        
        if (x == c || y == c)
            return steps;
        
        if (!visited[a][y]) {
            visited[a][y] = true;
            q.push(make_tuple(a, y, steps + 1));
        }
        if (!visited[x][b]) {
            visited[x][b] = true;
            q.push(make_tuple(x, b, steps + 1));
        }
        if (!visited[0][y]) {
            visited[0][y] = true;
            q.push(make_tuple(0, y, steps + 1));
        }
        if (!visited[x][0]) {
            visited[x][0] = true;
            q.push(make_tuple(x, 0, steps + 1));
        }
        int trans = min(x, b - y);
        int nx = x - trans, ny = y + trans;
        if (!visited[nx][ny]) {
            visited[nx][ny] = true;
            q.push(make_tuple(nx, ny, steps + 1));
        }
        trans = min(y, a - x);
        nx = x + trans, ny = y - trans;
        if (!visited[nx][ny]) {
            visited[nx][ny] = true;
            q.push(make_tuple(nx, ny, steps + 1));
        }
    }
    
    return -1;
}

int main() {
    int a, b, c;
    cin >> a >> b >> c;
    cout << bfs(a, b, c);
    return 0;
}
