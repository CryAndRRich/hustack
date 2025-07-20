 #include <vector>
#include <stack>
#include <climits>
#include <algorithm>
using namespace std;

class Solution {
public:
    long long dp(int i, vector<int>& nums, vector<int>& costs, vector<int>& nse, vector<int>& nge, vector<long long>& memo, int n) {
        if (i == n) return LLONG_MAX;
        if (memo[i] != -1) return memo[i];
        if (i == n - 1) return memo[i] = (i == 0 ? 0 : costs[i]);
        long long cost = (i == 0 ? 0 : costs[i]);
        long long left = dp(nse[i], nums, costs, nse, nge, memo, n);
        long long right = dp(nge[i], nums, costs, nse, nge, memo, n);
        return memo[i] = cost + min(left, right);
    }

    long long minCost(vector<int>& nums, vector<int>& costs) {
        int n = nums.size();
        vector<int> nse(n, n), nge(n, n);
        vector<long long> memo(n, -1);
        stack<int> max_st, min_st;

        for (int i = 0; i < n; ++i) {
            while (!max_st.empty() && nums[max_st.top()] <= nums[i]) {
                nge[max_st.top()] = i;
                max_st.pop();
            }
            while (!min_st.empty() && nums[min_st.top()] > nums[i]) {
                nse[min_st.top()] = i;
                min_st.pop();
            }
            max_st.push(i);
            min_st.push(i);
        }

        return dp(0, nums, costs, nse, nge, memo, n);
    }
};
