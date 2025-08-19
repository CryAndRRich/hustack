#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int minConnectedGroups(vector<vector<int>>& intervals, int k) {
        sort(intervals.begin(), intervals.end());
        vector<pair<long long,long long>> merged;
        for (auto &it : intervals) {
            long long s = it[0], e = it[1];
            if (merged.empty() || s > merged.back().second)
                merged.emplace_back(s, e);
            else
                merged.back().second = max(merged.back().second, e);
        }
        int m = merged.size();
        if (m == 0) return 0;
        int res = m;
        for (int i = 0; i < m; ++i) {
            int lo = i, hi = m - 1, best = i;
            while (lo <= hi) {
                int mid = lo + (hi - lo) / 2;
                if (merged[mid].first - merged[i].second <= (long long)k) {
                    best = mid;
                    lo = mid + 1;
                } else hi = mid - 1;
            }
            int reduction = best - i;
            res = min(res, m - reduction);
        }
        return res;
    }
};
