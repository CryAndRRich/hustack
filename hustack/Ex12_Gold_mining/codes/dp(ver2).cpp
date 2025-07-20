#include <iostream>
#include <vector>
#include <deque>
#include <algorithm>
using namespace std;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, L1, L2;
    cin >> n >> L1 >> L2;
    
    vector<int> a(n);
    for (int i = 0; i < n; i++){
        cin >> a[i];
    }
    
    vector<int> dp(n, 0);
    int ans = 0;
    deque<int> dq;
    
    for (int i = 0; i < n; i++){
        while (!dq.empty() && dq.front() < i - L2)
            dq.pop_front();
        
        if (i - L1 >= 0) {
            int j = i - L1;
            while (!dq.empty() && dp[dq.back()] <= dp[j])
                dq.pop_back();
            dq.push_back(j);
        }
        
        int best_prev = (!dq.empty()) ? dp[dq.front()] : 0;
        dp[i] = a[i] + best_prev;
        ans = max(ans, dp[i]);
    }
    
    cout << ans;
    return 0;
}
