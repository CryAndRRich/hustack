#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int n;
    vector<int> t;
    SegTree(int _n): n(1) {
        while (n < _n) n <<= 1;
        t.assign(2 * n, 0);
    }
    void update(int i, int v) {
        i += n;
        t[i] = max(t[i], v);
        for (i >>= 1; i; i >>= 1)
            t[i] = max(t[2 * i], t[2 * i+1]);
    }
    int query(int l, int r) {
        int res = 0;
        for (l += n, r += n; l <= r; l >>= 1, r >>= 1) {
            if (l & 1) res = max(res, t[l++]);
            if (!(r & 1)) res = max(res, t[r--]);
        }
        return res;
    }
};

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n; cin >> n;
    vector<int> a(n), b;
    for (int i = 0; i < n; i++){
        cin >> a[i];
        b.push_back(a[i]);
    }
    sort(b.begin(), b.end());
    b.erase(unique(b.begin(), b.end()), b.end());
    int m = b.size();
    SegTree st(m);
    int best = 0;
    for (int x : a) {
        int idx = int(lower_bound(b.begin(), b.end(), x) - b.begin());
        int cur = 1;
        if (idx > 0) cur = st.query(0, idx-1) + 1;
        st.update(idx, cur);
        best = max(best, cur);
    }
    cout << best;
    return 0;
}
