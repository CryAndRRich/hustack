#include <vector>
#include <stack>
#include <climits>
#include <algorithm>

class Solution {
public:
    long long minCost(vector<int>& nums, vector<int>& costs) {
        int n = nums.size();
        vector<int> nse(n, n), nge(n, n);
        vector<long long> dp(n, LLONG_MAX);
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
        dp[0] = 0;
        for (int i = 0; i < n; ++i) {
            for (int j : {nse[i], nge[i]}) {
                if (j < n) {
                    dp[j] = min(dp[j], dp[i] + costs[j]);
                }
            }
        }
        return dp[n - 1];
    }
};
