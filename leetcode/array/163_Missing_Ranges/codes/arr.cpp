class Solution {
public:
    vector<vector<int>> findMissingRanges(vector<int>& nums, int lower, int upper) {
        vector<vector<int>> res;
        nums.insert(nums.begin(), lower - 1);
        nums.push_back(upper + 1);
        for (int i = 1; i < nums.size(); ++i) {
            if (nums[i] - nums[i - 1] > 1)
                res.push_back({nums[i - 1] + 1, nums[i] - 1});
        }
        return res;
    }
};
