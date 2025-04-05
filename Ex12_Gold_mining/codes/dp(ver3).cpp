#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct SegmentTree {
    int size;
    vector<int> tree;
    
    SegmentTree(int n) {
        size = 1;
        while (size < n) size *= 2;
        tree.assign(2 * size, 0);
    }
    
    void update(int ind, int val) {
        ind += size;
        tree[ind] = val;
        while (ind > 1) {
            ind /= 2;
            tree[ind] = max(tree[2 * ind], tree[2 * ind + 1]);
        }
    }
    
    int query(int l, int r) {
        int res = 0;
        l += size;
        r += size;
        while (l <= r) {
            if (l % 2 == 1) {
                res = max(res, tree[l]);
                l++;
            }
            if (r % 2 == 0) {
                res = max(res, tree[r]);
                r--;
            }
            l /= 2;
            r /= 2;
        }
        return res;
    }
};

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
    SegmentTree seg(n);
    
    for (int i = 0; i < n; i++){
        int best_prev = 0;
        if (i - L1 >= 0) {
            int left = max(0, i - L2);
            int right = i - L1;
            best_prev = seg.query(left, right);
        }
        dp[i] = a[i] + best_prev;
        ans = max(ans, dp[i]);
        seg.update(i, dp[i]);
    }
    
    cout << ans;
    return 0;
}
