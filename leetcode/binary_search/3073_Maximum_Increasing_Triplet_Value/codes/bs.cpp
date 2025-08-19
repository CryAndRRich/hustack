#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int maximumTripletValue(vector<int>& nums) {
        int n = nums.size();
        vector<int> right(n);
        int maxVal = 0;
        for (int i = n - 1; i >= 0; i--) {
            right[i] = maxVal;
            maxVal = max(maxVal, nums[i]);
        }
        set<int> s;
        int ans = 0;
        for (int i = 0; i < n; i++) {
            if (!s.empty() && right[i] > nums[i]) {
                auto it = s.lower_bound(nums[i]);
                if (it != s.begin()) {
                    --it;
                    int greatestLeft = *it;
                    ans = max(ans, greatestLeft - nums[i] + right[i]);
                }
            }
            s.insert(nums[i]);
        }
        return ans;
    }
};
