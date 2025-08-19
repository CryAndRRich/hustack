#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int minimumDistance(int n, vector<vector<int>>& edges, int s, vector<int>& marked) {
        unordered_set<int> markSet(marked.begin(), marked.end());
        vector<vector<pair<int,int>>> adj(n);
        for (auto &e : edges) adj[e[0]].push_back({e[1], e[2]});
        
        vector<int> dist(n, INT_MAX);
        dist[s] = 0;
        priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
        pq.push({0, s});
        
        while (!pq.empty()) {
            auto [d, node] = pq.top();
            pq.pop();
            if (markSet.count(node)) return d;
            for (auto &[nei, w] : adj[node]) {
                int nd = d + w;
                if (nd < dist[nei]) {
                    dist[nei] = nd;
                    pq.push({nd, nei});
                }
            }
        }
        return -1;
    }
};
