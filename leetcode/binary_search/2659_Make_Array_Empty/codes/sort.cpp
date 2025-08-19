#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long countOperationsToEmptyArray(vector<int>& nums) {
        int N = nums.size();
        unordered_map<int, int> locs;
        for (int i = 0; i < N; ++i)
            locs[nums[i]] = i;

        vector<int> positions(N);
        iota(positions.begin(), positions.end(), 0);

        long long ans = 0;
        int i = 0;
        sort(nums.begin(), nums.end());
        for (int x : nums) {
            int r = locs[x];
            int j = lower_bound(positions.begin(), positions.end(), r) - positions.begin();
            ans += (j - i + positions.size()) % positions.size();
            positions.erase(positions.begin() + j);
            i = j;
        }
        return ans + N;
    }
};
