#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int makeArrayPositive(vector<int>& nums) {
        int n = nums.size();
        vector<long long> pref(n+1, 0);
        for (int i = 0; i < n; ++i) pref[i+1] = pref[i] + nums[i];
        int res = 0;
        int covered = -1;
        for (int end = 2; end < n; ++end) {
            bool need = false;
            for (int l = 3; l <= 5; ++l) {
                int start = end - (l - 1);
                if (start < 0) break;
                if (covered >= start && covered <= end) continue;
                if (pref[end+1] - pref[start] <= 0) { need = true; break; }
            }
            if (need) { res++; covered = end; }
        }
        return res;
    }
};
