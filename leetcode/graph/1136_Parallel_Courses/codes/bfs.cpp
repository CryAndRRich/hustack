#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    int minimumSemesters(int n, vector<vector<int>>& relations) {
        vector<vector<int>> graph(n + 1);
        vector<int> in_degree(n + 1, 0);

        for (auto& r : relations) {
            graph[r[0]].push_back(r[1]);
            in_degree[r[1]]++;
        }

        queue<int> q;
        for (int i = 1; i <= n; ++i)
            if (in_degree[i] == 0)
                q.push(i);

        int semesters = 0, taken = 0;

        while (!q.empty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int node = q.front(); q.pop();
                taken++;
                for (int nei : graph[node]) {
                    if (--in_degree[nei] == 0)
                        q.push(nei);
                }
            }
            semesters++;
        }

        return taken == n ? semesters : -1;
    }
};
