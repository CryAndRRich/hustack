#include <vector>
#include <stack>
#include <algorithm>
using namespace std;

class Solution {
public:
    int visibleMountains(vector<vector<int>>& peaks) {
        sort(peaks.begin(), peaks.end());

        stack<int> st;
        int end = INT_MIN;

        for (const auto& peak : peaks) {
            int left = peak[0] - peak[1];
            int right = peak[0] + peak[1];

            while (!st.empty() && st.top() >= left) {
                st.pop();
            }

            if (right > end) {
                st.push(left);
                end = right;
            }
        }

        return st.size();
    }
};
