#include <vector>
#include <queue>
#include <climits>
using namespace std;

class Solution {
public:
    int shortestDistance(vector<vector<int>>& grid) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> dist(m, vector<int>(n, 0));
        vector<vector<int>> reach(m, vector<int>(n, 0));
        int buildings = 0;

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (grid[i][j] == 1) {
                    buildings++;
                    bfs(grid, dist, reach, i, j);
                }

        int res = INT_MAX;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (grid[i][j] == 0 && reach[i][j] == buildings)
                    res = min(res, dist[i][j]);
        return res == INT_MAX ? -1 : res;
    }

    void bfs(vector<vector<int>>& grid, vector<vector<int>>& dist, vector<vector<int>>& reach, int x, int y) {
        int m = grid.size(), n = grid[0].size();
        vector<vector<bool>> visited(m, vector<bool>(n, false));
        queue<pair<int, int>> q;
        q.push({x, y});
        int level = 0;

        while (!q.empty()) {
            int sz = q.size();
            level++;
            for (int i = 0; i < sz; ++i) {
                auto [cx, cy] = q.front(); q.pop();
                for (auto [dx, dy] : vector<pair<int,int>>{{0,1},{1,0},{0,-1},{-1,0}}) {
                    int nx = cx + dx, ny = cy + dy;
                    if (nx >= 0 && ny >= 0 && nx < m && ny < n &&
                        !visited[nx][ny] && grid[nx][ny] == 0) {
                        visited[nx][ny] = true;
                        dist[nx][ny] += level;
                        reach[nx][ny]++;
                        q.push({nx, ny});
                    }
                }
            }
        }
    }
};
