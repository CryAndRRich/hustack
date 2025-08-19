#include <vector>
#include <stack>
using namespace std;

class Solution {
public:
    vector<vector<int>> seePeople(vector<vector<int>>& heights) {
        int r = heights.size(), c = heights[0].size();
        vector<vector<int>> res(r, vector<int>(c, 0));

        for (int i = 0; i < r; ++i) {
            stack<int> st;
            for (int j = c - 1; j >= 0; --j) {
                bool equal = false;
                while (!st.empty() && st.top() <= heights[i][j]) {
                    if (st.top() == heights[i][j]) equal = true;
                    st.pop();
                    res[i][j]++;
                }
                if (!st.empty() && !equal) res[i][j]++;
                st.push(heights[i][j]);
            }
        }

        for (int j = 0; j < c; ++j) {
            stack<int> st;
            for (int i = r - 1; i >= 0; --i) {
                bool equal = false;
                while (!st.empty() && st.top() <= heights[i][j]) {
                    if (st.top() == heights[i][j]) equal = true;
                    st.pop();
                    res[i][j]++;
                }
                if (!st.empty() && !equal) res[i][j]++;
                st.push(heights[i][j]);
            }
        }

        return res;
    }
};
