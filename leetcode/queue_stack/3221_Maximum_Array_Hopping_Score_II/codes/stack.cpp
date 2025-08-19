#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long maxScore(vector<int>& nums) {
        vector<pair<int,int>> stack;
        stack.push_back({nums.back(), (int)nums.size() - 1});
        for (int i = nums.size() - 2; i >= 0; --i) {
            if (nums[i] > stack.back().first)
                stack.push_back({nums[i], i});
        }
        long long res = 0;
        int j = 0;
        for (int k = stack.size() - 1; k >= 0; --k) {
            res += 1LL * (stack[k].second - j) * stack[k].first;
            j = stack[k].second;
        }
        return res;
    }
};
