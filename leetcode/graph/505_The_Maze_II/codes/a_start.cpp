#include <vector>
#include <queue>
#include <climits>
#include <cmath>

using namespace std;

class Solution {
public:
    int shortestDistance(vector<vector<int>>& maze, vector<int>& start, vector<int>& destination) {
        int m = maze.size(), n = maze[0].size();
        vector<vector<int>> dist(m, vector<int>(n, INT_MAX));
        dist[start[0]][start[1]] = 0;

        auto heuristic = [&](int x, int y) {
            return abs(x - destination[0]) + abs(y - destination[1]);
        };

        priority_queue<tuple<int, int, int, int>, vector<tuple<int, int, int, int>>, greater<>> pq;
        pq.emplace(heuristic(start[0], start[1]), 0, start[0], start[1]);

        vector<pair<int, int>> dirs = {{-1, 0},{1, 0},{0, -1},{0, 1}};

        while (!pq.empty()) {
            auto [f, d, x, y] = pq.top(); pq.pop();
            if (x == destination[0] && y == destination[1]) return d;

            for (auto [dx, dy] : dirs) {
                int nx = x, ny = y, steps = 0;
                while (nx + dx >= 0 && nx + dx < m && ny + dy >= 0 && ny + dy < n && maze[nx + dx][ny + dy] == 0) {
                    nx += dx;
                    ny += dy;
                    ++steps;
                }

                if (d + steps < dist[nx][ny]) {
                    dist[nx][ny] = d + steps;
                    int new_f = dist[nx][ny] + heuristic(nx, ny);
                    pq.emplace(new_f, dist[nx][ny], nx, ny);
                }
            }
        }

        return -1;
    }
};
