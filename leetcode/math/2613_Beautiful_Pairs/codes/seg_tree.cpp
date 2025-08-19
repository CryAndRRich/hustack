#include <bits/stdc++.h>
using namespace std;

struct Seg {
    vector<int> tree;
    int size;
    Seg(int n=0){ size = n; tree.assign(4*(n+5), -1); }
    int better(const vector<int> &v, int x, int y) {
        if (x < 0) return y;
        if (y < 0) return x;
        if (v[x] < v[y] || (v[x] == v[y] && y < x)) return y;
        return x;
    }
    int query(const vector<int> &v, int ind, int l, int r, int ql, int qr) {
        if (ql > qr) return -1;
        if (l >= ql && r <= qr) return tree[ind];
        int mid = (l + r) >> 1;
        int res = -1;
        if (ql <= mid) res = query(v, ind<<1, l, mid, ql, qr);
        if (qr > mid) res = better(v, res, query(v, (ind<<1)|1, mid+1, r, ql, qr));
        return res;
    }
    void update(const vector<int> &v, int ind, int l, int r, int pos, int val) {
        tree[ind] = better(v, tree[ind], val);
        if (l == r) return;
        int mid = (l + r) >> 1;
        if (pos <= mid) update(v, ind<<1, l, mid, pos, val);
        else update(v, (ind<<1)|1, mid+1, r, pos, val);
    }
};

class Solution {
public:
    vector<int> beautifulPair(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size();
        vector<int> p1(n), p2(n);
        for (int i = 0; i < n; ++i) { p1[i] = nums2[i] + nums1[i]; p2[i] = nums2[i] - nums1[i]; }
        int maxCoord = 0;
        for (int v : nums1) maxCoord = max(maxCoord, v);
        Seg seg(maxCoord+2);
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b){ return nums2[a] < nums2[b]; });
        int ans0 = n, ans1 = n;
        long long d = LLONG_MAX;
        for (int k = 0; k < n; ++k) {
            int i = idx[k];
            int x = nums1[i];
            if (x >= 0) {
                int i1 = seg.query(p1, 1, 0, maxCoord+1, 0, x);
                if (i1 >= 0) {
                    long long may = (long long)p1[i] - (long long)p1[i1];
                    if (may < d || (may == d && (min(i,i1) < ans0 || (min(i,i1) == ans0 && max(i,i1) < ans1)))) {
                        d = may; ans0 = min(i,i1); ans1 = max(i,i1);
                    }
                }
            }
            int i2 = seg.query(p2, 1, 0, maxCoord+1, x, maxCoord+1);
            if (i2 >= 0) {
                long long may = (long long)p2[i] - (long long)p2[i2];
                if (may < d || (may == d && (min(i,i2) < ans0 || (min(i,i2) == ans0 && max(i,i2) < ans1)))) {
                    d = may; ans0 = min(i,i2); ans1 = max(i,i2);
                }
            }
            seg.update(p1, 1, 0, maxCoord+1, x, i);
            seg.update(p2, 1, 0, maxCoord+1, x, i);
        }
        return {ans0, ans1};
    }
};
