#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

bool cmp(const vector<int> &a, const vector<int> &b) {
    if(a[1] == b[1])
        return a[0] < b[0];
    return a[1] < b[1];
}

int binarySearch(const vector<vector<int>> &segments, int left, int right, int target) {
    int ans = -1;
    while(left <= right) {
        int mid = left + (right - left) / 2;
        if(segments[mid][1] < target) {
            ans = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return ans;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n;
    cin >> n;
    vector<vector<int>> segments(n, vector<int>(2));
    
    for (int i = 0; i < n; i++){
        cin >> segments[i][0] >> segments[i][1];
    }
    
    sort(segments.begin(), segments.end(), cmp);
    
    vector<int> dp(n + 1, 0);
    dp[0] = 0;
    for (int i = 0; i < n; i++){
        int ind = binarySearch(segments, 0, i - 1, segments[i][0]);
        int candidate = 1;
        if (ind != -1) {
            candidate += dp[ind + 1];
        }
        dp[i + 1] = max(dp[i], candidate);
    }
    
    cout << dp[n];
    return 0;
}
