#include <vector>
#include <queue>
#include <utility>
using namespace std;

class Solution {
public:
    int minCostToSupplyWater(int n, vector<int>& wells, vector<vector<int>>& pipes) {
        vector<vector<pair<int, int>>> graph(n + 1);

        for (int i = 0; i < n; ++i) {
            graph[0].emplace_back(i + 1, wells[i]);
            graph[i + 1].emplace_back(0, wells[i]);
        }

        for (auto& p : pipes) {
            graph[p[0]].emplace_back(p[1], p[2]);
            graph[p[1]].emplace_back(p[0], p[2]);
        }

        vector<bool> visited(n + 1, false);
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
        pq.emplace(0, 0);

        int total = 0;
        while (!pq.empty()) {
            auto [cost, u] = pq.top(); pq.pop();
            if (visited[u]) continue;
            visited[u] = true;
            total += cost;
            for (auto& [v, w] : graph[u]) {
                if (!visited[v]) {
                    pq.emplace(w, v);
                }
            }
        }

        return total;
    }
};
