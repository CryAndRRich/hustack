#include <vector>
#include <queue>
#include <climits>
using namespace std;

class Solution {
public:
    int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
        int m = maze.size(), n = maze[0].size();
        vector<vector<int>> dist(m, vector<int>(n, INT_MAX));
        dist[start[0]][start[1]] = 0;
        priority_queue<vector<int>, vector<vector<int>>, greater<>> pq;
        pq.push({0, start[0], start[1]});
        
        vector<vector<int>> dirs = {{-1, 0},{1, 0},{0, -1},{0, 1}};
        
        while (!pq.empty()) {
            vector<int> cur = pq.top(); 
            int d = cur[0], x = cur[1], y = cur[2];
            pq.pop();
            if (x == destination[0] && y == destination[1]) return d;
            for (auto& dir : dirs) {
                int nx = x, ny = y, steps = 0;
                while (nx + dir[0] >= 0 && nx + dir[0] < m &&
                       ny + dir[1] >= 0 && ny + dir[1] < n &&
                       maze[nx + dir[0]][ny + dir[1]] == 0) {
                    nx += dir[0];
                    ny += dir[1];
                    steps++;
                }
                if (d + steps < dist[nx][ny]) {
                    dist[nx][ny] = d + steps;
                    pq.push({d + steps, nx, ny});
                }
            }
        }
        return -1;
    }
};
