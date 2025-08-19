#include <bits/stdc++.h>
using namespace std;

struct SegmentTree {
    int n;
    vector<int> arr;
    SegmentTree(int n) : n(n), arr(2 * n, -1) {}
    void update(int idx, int val) {
        idx += n;
        arr[idx] = val;
        while (idx > 1) {
            idx /= 2;
            arr[idx] = max(arr[2 * idx], arr[2 * idx + 1]);
        }
    }
    int query(int l, int r) {
        if (l > r) return -1;
        l += n; r += n;
        int res = -1;
        while (l <= r) {
            if (l & 1) res = max(res, arr[l++]);
            if (!(r & 1)) res = max(res, arr[r--]);
            l /= 2; r /= 2;
        }
        return res;
    }
};

class Solution {
public:
    int maxProfit(vector<int>& prices, vector<int>& profits) {
        int n = prices.size();
        set<int> s(prices.begin(), prices.end());
        unordered_map<int,int> mp;
        int idx = 0;
        for (int p : s) mp[p] = idx++;
        int m = mp.size();
        vector<int> maxleft(n, -1), maxright(n, -1), valleft(m, -1), valright(m, -1);
        SegmentTree segleft(m), segright(m);
        for (int i = 0; i < n; i++) {
            int id = mp[prices[i]];
            maxleft[i] = segleft.query(0, id - 1);
            if (profits[i] > valleft[id]) {
                valleft[id] = profits[i];
                segleft.update(id, profits[i]);
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            int id = mp[prices[i]];
            maxright[i] = segright.query(id + 1, m - 1);
            if (profits[i] > valright[id]) {
                valright[id] = profits[i];
                segright.update(id, profits[i]);
            }
        }
        int ans = -1;
        for (int i = 0; i < n; i++) {
            if (maxleft[i] >= 0 && maxright[i] >= 0) {
                ans = max(ans, profits[i] + maxleft[i] + maxright[i]);
            }
        }
        return ans;
    }
};
