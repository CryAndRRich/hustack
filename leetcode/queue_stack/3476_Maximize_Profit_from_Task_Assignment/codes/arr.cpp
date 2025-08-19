#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long maxProfit(vector<int>& workers, vector<vector<int>>& tasks) {
        unordered_map<int, priority_queue<int>> mp;
        for (auto &t : tasks) {
            mp[t[0]].push(t[1]);
        }
        long long total = 0;
        for (int w : workers) {
            if (mp.count(w) && !mp[w].empty()) {
                total += mp[w].top();
                mp[w].pop();
            }
        }
        int maxExtra = 0;
        for (auto &kv : mp) {
            if (!kv.second.empty()) {
                maxExtra = max(maxExtra, kv.second.top());
            }
        }
        total += maxExtra;
        return total;
    }
};
