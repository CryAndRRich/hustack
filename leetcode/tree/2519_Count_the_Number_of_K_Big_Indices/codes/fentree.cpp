#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>

using namespace std;

class FenwickTree {
public:
    vector<int> tree;
    FenwickTree(int size) {
        tree.resize(size + 2);
    }

    void update(int i, int delta) {
        while (i < tree.size()) {
            tree[i] += delta;
            i += (i & -i);
        }
    }

    int query(int i) {
        int res = 0;
        while (i > 0) {
            res += tree[i];
            i -= (i & -i);
        }
        return res;
    }
};

class Solution {
public:
    int kBigIndices(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> sorted = nums;
        sort(sorted.begin(), sorted.end());
        unordered_map<int, int> compress;
        int id = 1;
        for (int num : sorted)
            if (compress.find(num) == compress.end())
                compress[num] = id++;

        vector<int> comp(n);
        for (int i = 0; i < n; ++i)
            comp[i] = compress[nums[i]];

        vector<int> left(n), right(n);
        FenwickTree bit(id);

        for (int i = 0; i < n; ++i) {
            left[i] = bit.query(comp[i] - 1);
            bit.update(comp[i], 1);
        }

        bit = FenwickTree(id);
        for (int i = n - 1; i >= 0; --i) {
            right[i] = bit.query(comp[i] - 1);
            bit.update(comp[i], 1);
        }

        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (left[i] >= k && right[i] >= k)
                ++count;
        }
        return count;
    }
};
