#include <bits/stdc++.h>
using namespace std;

class BIT {
public:
    vector<int> bit;
    int n;
    BIT(int size) : n(size), bit(size + 2, 0) {}

    void update(int i, int delta) {
        i++;
        while (i <= n + 1) {
            bit[i] += delta;
            i += i & -i;
        }
    }

    int query(int i) {
        i++;
        int res = 0;
        while (i) {
            res += bit[i];
            i -= i & -i;
        }
        return res;
    }

    int range_query(int l, int r) {
        return query(r) - query(l - 1);
    }
};

class Solution {
public:
    long long countOperationsToEmptyArray(vector<int>& nums) {
        int n = nums.size();
        unordered_map<int, int> index_map;
        for (int i = 0; i < n; i++) index_map[nums[i]] = i;

        BIT bit(n);
        for (int i = 0; i < n; i++) bit.update(i, 1);

        vector<int> sorted_nums = nums;
        sort(sorted_nums.begin(), sorted_nums.end());

        long long total = 0;
        int prev_pos = 0;
        for (int num : sorted_nums) {
            int pos = index_map[num];
            int ops = 0;
            if (pos >= prev_pos) {
                ops = bit.range_query(prev_pos, pos);
            } else {
                ops = bit.range_query(prev_pos, n - 1) + bit.range_query(0, pos);
            }
            total += ops;
            bit.update(pos, -1);
            prev_pos = pos;
        }
        return total;
    }
};
