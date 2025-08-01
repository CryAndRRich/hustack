#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minimumSemesters(int n, vector<vector<int>>& relations) {
        vector<vector<int>> graph(n + 1);
        for (auto& r : relations)
            graph[r[0]].push_back(r[1]);

        vector<int> visited(n + 1, 0);
        vector<int> dp(n + 1, 0);

        function<int(int)> dfs = [&](int node) -> int {
            if (visited[node] == 1) return -1; 
            if (visited[node] == 2) return dp[node];

            visited[node] = 1;
            int maxDepth = 1;
            for (int nei : graph[node]) {
                int res = dfs(nei);
                if (res == -1) return -1;
                maxDepth = max(maxDepth, res + 1);
            }
            visited[node] = 2;
            dp[node] = maxDepth;
            return dp[node];
        };

        int res = 0;
        for (int i = 1; i <= n; ++i) {
            int r = dfs(i);
            if (r == -1) return -1;
            res = max(res, r);
        }
        return res;
    }
};
