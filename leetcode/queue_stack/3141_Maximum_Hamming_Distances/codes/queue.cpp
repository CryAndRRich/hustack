#include <bits/stdc++.h>
using namespace std;
class Solution {
public:
    vector<int> maxHammingDistances(vector<int>& nums, int m) {
        int n = 1 << m;
        vector<int> dist(n, -1);
        queue<int> q;
        for (int x : nums) {
            dist[x] = 0;
            q.push(x);
        }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int b = 0; b < m; b++) {
                int v = u ^ (1 << b);
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        int mask = n - 1;
        vector<int> ans;
        ans.reserve(nums.size());
        for (int x : nums) {
            ans.push_back(m - dist[mask ^ x]);
        }
        return ans;
    }
};
