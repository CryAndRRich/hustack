#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct FenTree {
    int size;
    vector<int> tree; 
    
    FenTree(int n) {
        size = n;
        tree.assign(n + 1, 0);
    }
    
    void update(int index, int val) {
        while(index <= size) {
            tree[index] = max(tree[index], val);
            index += index & (-index);
        }
    }
    
    int query_prefix(int index) {
        int res = 0;
        while(index > 0) {
            res = max(res, tree[index]);
            index -= index & (-index);
        }
        return res;
    }
    
    int query_range(int L, int R, const vector<int>& dp) {
        int res = 0;
        while(R >= L) {
            int start = R - (R & -R) + 1;
            if(start >= L) {
                res = max(res, tree[R]);
                R = start - 1;
            } else {
                res = max(res, dp[R]);
                R--;
            }
        }
        return res;
    }
};

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n, L1, L2;
    cin >> n >> L1 >> L2;
    vector<int> a(n + 1), dp(n + 1, 0);
    for (int i = 1; i <= n; i++){
        cin >> a[i];
    }
    
    int ans = 0;
    FenTree ft(n);
    
    for (int i = 1; i <= n; i++){
        int best_prev = 0;
        if(i - L1 >= 1) {
            int left = (i - L2 < 1) ? 1 : i - L2;
            int right = i - L1;
            best_prev = ft.query_range(left, right, dp);
        }
        dp[i] = a[i] + best_prev;
        ans = max(ans, dp[i]);
        ft.update(i, dp[i]);
    }
    
    cout << ans;
    return 0;
}
