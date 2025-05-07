#include <bits/stdc++.h>
using namespace std;

int largestRectangleHistogram(const vector<int>& h) {
    stack<int> st;
    int maxArea = 0;
    int m = h.size();
    vector<int> H = h;
    H.push_back(0);
    for (int i = 0; i <= m; ++i) {
        while (!st.empty() && H[st.top()] > H[i]) {
            int height = H[st.top()]; st.pop();
            int width = st.empty() ? i : (i - st.top() - 1);
            maxArea = max(maxArea, height * width);
        }
        st.push(i);
    }
    return maxArea;
}

int maximalRectangle(const vector<vector<int>>& A, int n, int m) {
    vector<int> heights(m, 0);
    int best = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            heights[j] = (A[i][j] == 1 ? heights[j] + 1 : 0);
        }
        best = max(best, largestRectangleHistogram(heights));
    }
    return best;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;
    vector<vector<int>> A(n, vector<int>(m));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            cin >> A[i][j];

    cout << maximalRectangle(A, n, m);
    return 0;
}
