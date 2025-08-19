#include <vector>
#include <cmath>
#include <numeric>
using namespace std;

class Solution {
    int get(const vector<vector<int>>& st, int l, int r) {
        int j = log2(r - l + 1);
        return gcd(st[l][j], st[r - (1 << j) + 1][j]);
    }

public:
    long long maxGcdSum(vector<int>& nums, int k) {
        int n = nums.size();
        vector<vector<int>> st(n, vector<int>(20));
        vector<long long> prefix(n + 1);

        for (int i = 0; i < n; ++i) {
            prefix[i + 1] = prefix[i] + nums[i];
            st[i][0] = nums[i];
        }

        for (int j = 1; j < 20; ++j) {
            for (int i = 0; i + (1 << j) <= n; ++i) {
                st[i][j] = gcd(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
            }
        }

        long long res = 0;
        for (int i = 0; i < n; ++i) {
            int want = nums[i], left = i + 1;
            while (left <= n && (prefix[n] - prefix[i]) * want > res) {
                int lo = left, hi = n - 1;
                while (lo <= hi) {
                    int mid = (lo + hi) / 2;
                    if (get(st, i, mid) == want) lo = mid + 1;
                    else hi = mid - 1;
                }
                if (hi - i + 1 >= k)
                    res = max(res, (prefix[hi + 1] - prefix[i]) * want);
                if (lo >= n) break;
                want = gcd(want, nums[lo]);
                left = lo + 1;
            }
        }
        return res;
    }
};
