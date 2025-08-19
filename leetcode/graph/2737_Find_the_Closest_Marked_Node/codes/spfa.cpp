#include <vector>
#include <queue>
#include <climits>
using namespace std;

class Solution {
public:
    int minimumDistance(int n, vector<vector<int>>& edges, int s, vector<int>& marked) {
        vector<vector<pair<int,int>>> graph(n);
        for (auto &e : edges) graph[e[0]].push_back({e[1], e[2]});
        vector<int> dist(n, INT_MAX);
        dist[s] = 0;
        queue<int> q;
        vector<bool> inQueue(n, false);
        q.push(s);
        inQueue[s] = true;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            inQueue[u] = false;
            for (auto &[v, w] : graph[u]) {
                if (dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                    if (!inQueue[v]) {
                        q.push(v);
                        inQueue[v] = true;
                    }
                }
            }
        }
        int ans = INT_MAX;
        for (int m : marked) ans = min(ans, dist[m]);
        return ans == INT_MAX ? -1 : ans;
    }
};
