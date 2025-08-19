#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int maxIntersectionCount(vector<int>& y) {
        int res = 0;
        vector<pair<long long, long long>> candles;
        for (int i = 0; i + 1 < (int)y.size(); ++i) {
            long long l = 2LL * y[i] + (i ? (y[i] < y[i + 1] ? 1 : -1) : 0);
            long long r = 2LL * y[i + 1];
            candles.push_back({min(l, r), max(l, r)});
        }
        sort(candles.begin(), candles.end());
        priority_queue<long long, vector<long long>, greater<long long>> pq;
        for (auto [l, r] : candles) {
            pq.push(r);
            while (!pq.empty() && pq.top() < l)
                pq.pop();
            res = max(res, (int)pq.size());
        }
        return res;
    }
};
