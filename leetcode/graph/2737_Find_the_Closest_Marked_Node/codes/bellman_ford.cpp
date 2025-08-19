#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int minimumDistance(int n, vector<vector<int>>& edges, int s, vector<int>& marked) {
        vector<int> dist(n, INT_MAX);
        dist[s] = 0;

        for (int iter = 0; iter < n - 1; iter++) {
            for (auto& edge : edges) {
                int u = edge[0];
                int v = edge[1];
                int w = edge[2];
                if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                }
            }
        }

        int minDist = INT_MAX;
        for (int m : marked) {
            if (dist[m] < minDist) minDist = dist[m];
        }

        return minDist == INT_MAX ? -1 : minDist;
    }
};
