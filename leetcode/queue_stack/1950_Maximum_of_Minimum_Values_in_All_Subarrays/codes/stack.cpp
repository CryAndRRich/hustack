#include <vector>
#include <stack>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<int> findMaximums(vector<int>& nums) {
        int n = nums.size();
        vector<int> left(n, -1), right(n, n), res(n, 0);
        stack<int> st;

        for (int i = 0; i < n; ++i) {
            while (!st.empty() && nums[st.top()] >= nums[i])
                st.pop();
            if (!st.empty()) left[i] = st.top();
            st.push(i);
        }

        st = stack<int>();
        for (int i = n - 1; i >= 0; --i) {
            while (!st.empty() && nums[st.top()] >= nums[i])
                st.pop();
            if (!st.empty()) right[i] = st.top();
            st.push(i);
        }

        for (int i = 0; i < n; ++i) {
            int len = right[i] - left[i] - 1;
            res[len - 1] = max(res[len - 1], nums[i]);
        }

        for (int i = n - 2; i >= 0; --i) {
            res[i] = max(res[i], res[i + 1]);
        }

        return res;
    }
};
