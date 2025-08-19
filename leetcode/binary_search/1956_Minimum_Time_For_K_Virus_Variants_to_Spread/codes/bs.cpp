#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int minDayskVariants(vector<vector<int>>& points, int k) {
        int lo = 0, hi = 1'000'000'000;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (check(points, k, mid))
                hi = mid;
            else
                lo = mid + 1;
        }
        return lo;
    }

private:
    bool check(vector<vector<int>>& points, int k, int day) {
        map<int, map<int, int>> lines;
        for (auto& p : points) {
            int x = p[0], y = p[1];
            lines[x + y - day][y - x - day]++;
            lines[x + y - day][y - x + day + 1]--;
            lines[x + y + day + 1][y - x - day]--;
            lines[x + y + day + 1][y - x + day + 1]++;
        }

        map<int, int> line;
        for (auto& [xx, row] : lines) {
            for (auto& [yy, v] : row)
                line[yy] += v;
            int prefix = 0;
            for (auto& [yy, val] : line) {
                prefix += val;
                bool condition = (xx - yy) % 2 == 0 || !lines.count(xx + 1) || 
                                 (prefix + line[yy + 1] >= k);
                if (prefix >= k && condition)
                    return true;
            }
        }
        return false;
    }
};
