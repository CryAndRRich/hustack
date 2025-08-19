#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int kEmptySlots(vector<int>& flowers, int k) {
        if (flowers.empty() || k + 2 > (int)flowers.size()) return -1;
        int n = flowers.size();
        vector<int> days(n);
        for (int i = 0; i < n; i++) {
            days[flowers[i] - 1] = i + 1;
        }

        int res = INT_MAX;
        deque<int> dayMin;

        for (int i = 1; i <= k; i++) {
            while (!dayMin.empty() && days[dayMin.back()] > days[i]) {
                dayMin.pop_back();
            }
            dayMin.push_back(i);
        }

        for (int i = k + 1; i < n; i++) {
            int maxDay = max(days[i - k - 1], days[i]);
            if (k == 0) {
                res = min(res, maxDay);
            } else if (maxDay < days[dayMin.front()]) {
                res = min(res, maxDay);
            }

            if (!dayMin.empty() && dayMin.front() == i - k) {
                dayMin.pop_front();
            }
            while (!dayMin.empty() && days[dayMin.back()] > days[i]) {
                dayMin.pop_back();
            }
            dayMin.push_back(i);
        }

        return (res == INT_MAX) ? -1 : res;
    }
};
