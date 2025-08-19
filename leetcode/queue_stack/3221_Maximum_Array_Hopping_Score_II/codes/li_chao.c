#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef long long ll;

typedef struct { 
    ll m, b; 
} Line;

Line *st;

ll eval(Line ln, int x){ 
    return ln.m * x + ln.b; 
}

void add_line(int idx, int l, int r, Line nw){
    Line cur = st[idx];
    int mid = (l + r) >> 1;
    int left = eval(nw, l) > eval(cur, l);
    int m = eval(nw, mid) > eval(cur, mid);
    if(m){ Line tmp = st[idx]; st[idx] = nw; nw = tmp; cur = st[idx]; }
    if(l == r) return;
    if(left != m) add_line(idx<<1, l, mid, nw);
    else add_line(idx<<1|1, mid + 1, r, nw);
}
ll query_line(int idx, int l, int r, int x){
    ll res = eval(st[idx], x);
    if(l == r) return res;
    int mid = (l + r) >> 1;
    if(x <= mid){
        ll t = query_line(idx<<1, l, mid, x);
        return t > res ? t : res;
    } else {
        ll t = query_line(idx<<1|1, mid + 1, r, x);
        return t > res ? t : res;
    }
}
long long maxScore(int* nums, int numsSize) {
    if(numsSize == 0) return 0;
    int n = numsSize;
    int size = 4 * (n + 5);
    st = (Line*) malloc(size * sizeof(Line));
    for(int i = 0; i < size; ++i){ 
        st[i].m = 0;
        st[i].b = LLONG_MIN/4; 
    }

    ll *dp = (ll*) calloc(n, sizeof(ll));
    add_line(1, 0, n - 1, (Line){ - (ll)nums[n - 1], (ll)(n - 1) * nums[n - 1] + dp[n - 1] });
    for(int i = n - 2; i >= 0; --i){
        ll best = query_line(1, 0, n - 1, i);
        dp[i] = best;
        add_line(1, 0, n - 1, (Line){ - (ll)nums[i], (ll)i * nums[i] + dp[i] });
    }
    ll ans = dp[0];
    free(st);
    free(dp);
    return ans;
}
