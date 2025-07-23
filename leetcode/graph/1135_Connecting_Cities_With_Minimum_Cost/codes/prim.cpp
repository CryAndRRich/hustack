#include <vector>
#include <queue>
#include <utility>

using namespace std;

class Solution {
public:
    int minimumCost(int n, vector<vector<int>>& connections) {
        vector<vector<pair<int, int>>> graph(n + 1);
        for (auto& con : connections) {
            graph[con[0]].emplace_back(con[1], con[2]);
            graph[con[1]].emplace_back(con[0], con[2]);
        }

        vector<bool> visited(n + 1, false);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        pq.emplace(0, 1);

        int total = 0, count = 0;
        while (!pq.empty() && count < n) {
            auto [cost, node] = pq.top(); pq.pop();
            if (visited[node]) continue;
            visited[node] = true;
            total += cost;
            count++;
            for (auto& [nei, w] : graph[node]) {
                if (!visited[nei])
                    pq.emplace(w, nei);
            }
        }

        return count == n ? total : -1;
    }
};
