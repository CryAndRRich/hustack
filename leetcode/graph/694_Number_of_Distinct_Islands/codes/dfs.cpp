class Solution {
public:
    int numDistinctIslands(vector<vector<int>>& grid) {
        set<vector<pair<int,int>>> shapes;
        int m = grid.size(), n = grid[0].size();

        function<void(int, int, int, int, vector<pair<int,int>>&)> dfs = [&](int r0, int c0, int r, int c, vector<pair<int,int>>& shape) {
            if (r < 0 || r >= m || c < 0 || c >= n || grid[r][c] == 0) return;
            grid[r][c] = 0;
            shape.emplace_back(r - r0, c - c0);
            dfs(r0, c0, r+1, c, shape);
            dfs(r0, c0, r-1, c, shape);
            dfs(r0, c0, r, c+1, shape);
            dfs(r0, c0, r, c-1, shape);
        };

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (grid[i][j] == 1) {
                    vector<pair<int,int>> shape;
                    dfs(i, j, i, j, shape);
                    shapes.insert(shape);
                }

        return shapes.size();
    }
};
