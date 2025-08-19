#include <bits/stdc++.h>
using namespace std;

using ll = long long;

struct Line { 
    ll m, b; 
    inline ll eval(ll x) const { 
        return m * x + b; 
    } 
};

struct Node {
    Line ln;
    Node *l = nullptr, *r = nullptr;
    Node(Line v): ln(v) {}
};

void add(Node*& node, int s, int e, Line nw) {
    if(!node){ 
        node = new Node(nw); 
        return; 
    }
    int mid = (s + e) >> 1;
    bool lef = nw.eval(s) > node -> ln.eval(s);
    bool m = nw.eval(mid) > node -> ln.eval(mid);
    if(m) swap(nw, node -> ln);
    if(s == e) return;
    if(lef != m) add(node -> l, s, mid, nw);
    else add(node -> r, mid+1, e, nw);
}
ll query(Node* node, int s, int e, int x) {
    if(!node) return LLONG_MIN/4;
    ll res = node -> ln.eval(x);
    if(s==e) return res;
    int mid = (s + e) >> 1;
    if(x <= mid) return max(res, query(node -> l, s, mid, x));
    else return max(res, query(node -> r, mid + 1, e, x));
}
class Solution {
public:
    long long maxScore(vector<int>& nums) {
        int n = nums.size();
        if(n == 0) return 0;

        vector<ll> dp(n, 0);
        Node* root = nullptr;
        add(root, 0, n - 1, Line{ - (ll)nums[n - 1], (ll)(n - 1) * nums[n - 1] + dp[n - 1] });
        for(int i = n - 2; i >= 0; --i) {
            ll best = query(root, 0, n - 1, i);
            dp[i] = best;
            add(root, 0, n - 1, Line{ - (ll)nums[i], (ll)i * nums[i] + dp[i] });
        }
        return dp[0];
    }
};
