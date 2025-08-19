#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int maximumTripletValue(vector<int>& nums) {
        int n = nums.size();
        vector<int> max_right(n, 0);
        int right_max = 0;

        for (int k = n - 1; k >= 0; --k) {
            if (nums[k] >= right_max) {
                right_max = nums[k];
            } else {
                max_right[k] = right_max;
            }
        }

        int max_triplet = 0;
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);

        sort(idx.begin(), idx.end(), [&](int a, int b) {
            if (nums[a] == nums[b]) return a > b;
            return nums[a] < nums[b];
        });

        vector<int> stack;
        for (int j : idx) {
            while (!stack.empty() && stack.back() > j) {
                stack.pop_back();
            }
            if (!stack.empty() && max_right[j] >= 0) {
                max_triplet = max(max_triplet, nums[stack.back()] - nums[j] + max_right[j]);
            }
            stack.push_back(j);
        }

        return max_triplet;
    }
};
