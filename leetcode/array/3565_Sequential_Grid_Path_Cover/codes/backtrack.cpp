#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int m, n, total, K;
    vector<vector<int>> grid;
    vector<vector<bool>> visited;
    vector<pair<int,int>> result;
    int dirs[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

    vector<vector<int>> findPath(vector<vector<int>>& g, int k) {
        grid = g;
        m = grid.size();
        n = grid[0].size();
        total = m * n;
        K = k;
        visited.assign(m, vector<bool>(n, false));
        vector<pair<int,int>> starts;
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) if (grid[i][j] == 0 || grid[i][j] == 1) starts.push_back({i,j});
        for (auto [sx, sy] : starts) {
            for (int i = 0; i < m; ++i) fill(visited[i].begin(), visited[i].end(), false);
            result.clear();
            if (backtrack(sx, sy, 1, 1)) {
                vector<vector<int>> ans;
                for (auto &p : result) ans.push_back({p.first, p.second});
                return ans;
            }
        }
        return {};
    }

    bool backtrack(int x, int y, int count, int next) {
        if (visited[x][y]) return false;
        if (grid[x][y] != 0 && grid[x][y] != next) return false;
        visited[x][y] = true;
        result.push_back({x, y});
        int newNext = (grid[x][y] == next ? next + 1 : next);
        if (count == total) {
            if (newNext == K + 1) return true;
            visited[x][y] = false;
            result.pop_back();
            return false;
        }
        for (int t = 0; t < 4; ++t) {
            int nx = x + dirs[t][0], ny = y + dirs[t][1];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                if (backtrack(nx, ny, count + 1, newNext)) return true;
            }
        }
        visited[x][y] = false;
        result.pop_back();
        return false;
    }
};
