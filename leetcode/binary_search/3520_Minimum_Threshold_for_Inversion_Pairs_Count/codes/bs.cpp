#include <bits/stdc++.h>
using namespace std;

struct Fenwick {
    int n;
    vector<long long> bit;
    Fenwick(int n): n(n), bit(n+1, 0) {}
    void update(int idx, int delta) {
        int i = idx + 1;
        while (i <= n) {
            bit[i] += delta;
            i += i & -i;
        }
    }
    long long query(int idx) {
        if (idx < 0) return 0;
        long long s = 0;
        int i = idx + 1;
        while (i > 0) {
            s += bit[i];
            i -= i & -i;
        }
        return s;
    }
    long long range_query(int l, int r) {
        if (l > r) return 0;
        return query(r) - query(l - 1);
    }
};

class Solution {
public:
    int minThreshold(vector<int>& nums, int k) {
        if (nums.empty()) return -1;
        vector<int> vals = nums;
        sort(vals.begin(), vals.end());
        vals.erase(unique(vals.begin(), vals.end()), vals.end());
        int m = vals.size();

        auto check = [&](int th) {
            Fenwick fw(m);
            long long cnt = 0;
            for (int num : nums) {
                int idxL = (int)(upper_bound(vals.begin(), vals.end(), num) - vals.begin()); 
                int idxR = (int)(upper_bound(vals.begin(), vals.end(), num + th) - vals.begin()) - 1; 
                if (idxL <= idxR) {
                    cnt += fw.range_query(idxL, idxR);
                    if (cnt >= k) return true;
                }
                int idxNum = (int)(lower_bound(vals.begin(), vals.end(), num) - vals.begin());
                fw.update(idxNum, 1);
            }
            return cnt >= k;
        };

        int lo = 0, hi = *max_element(nums.begin(), nums.end()) - *min_element(nums.begin(), nums.end());
        int ans = -1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (check(mid)) {
                ans = mid;
                hi = mid - 1;
            } else lo = mid + 1;
        }
        return ans;
    }
};
