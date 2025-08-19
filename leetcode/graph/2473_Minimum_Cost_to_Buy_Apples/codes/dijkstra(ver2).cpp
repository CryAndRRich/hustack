#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<long long> minCost(int n, vector<vector<int>>& roads, vector<int>& appleCost, int k) {
        vector<vector<pair<int, int>>> graph(n);
        for (auto& road : roads) {
            int a = road[0] - 1, b = road[1] - 1, c = road[2];
            graph[a].push_back({b, c});
            graph[b].push_back({a, c});
        }

        vector<long long> result(appleCost.begin(), appleCost.end());
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;
        for (int i = 0; i < n; i++) pq.push({result[i], i});

        while (!pq.empty()) {
            auto [cost, city] = pq.top(); pq.pop();
            if (cost > result[city]) continue;
            for (auto [nb, w] : graph[city]) {
                long long newCost = result[city] + (long long)(k + 1) * w;
                if (newCost < result[nb]) {
                    result[nb] = newCost;
                    pq.push({newCost, nb});
                }
            }
        }
        return result;
    }
};
