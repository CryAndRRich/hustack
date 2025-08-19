#include <bits/stdc++.h>
using namespace std;

const int MOD = 1e9 + 7;
const int RM = (MOD + 1) / 2;

int mul(long long x, long long y) {
    return (x * y) % MOD;
}

int sqr(int x) {
    return mul(x, x);
}

int add(int x, int y) {
    x += y;
    if (x >= MOD) x -= MOD;
    return x;
}

int sub(int x, int y) {
    x -= y;
    if (x < 0) x += MOD;
    return x;
}

int C2(int n) {
    return mul(mul(n, n - 1), RM);
}

int count(vector<int>& v, vector<int>& c, bool countIfEqual) {
    int n = v.size(), m = c.size();
    vector<int> t(m);
    int R2 = 0, RT = 0, R2T = 0, r = 0;

    for (int i = 0; i < m; ++i) R2 = add(R2, sqr(c[i]));

    for (int i = 0; i < n; ++i) {
        int x = v[i];
        int rx = c[x] - t[x];
        int r2 = sub(R2, sqr(rx));
        int rt = sub(RT, mul(rx, t[x]));
        int r2t = sub(R2T, mul(sqr(rx), t[x]));
        int p = n - i - rx;
        int sumt = i - t[x];

        int temp = mul(sub(sqr(p), r2), sumt);
        temp = sub(temp, mul(2LL * p % MOD, rt));
        temp = add(temp, mul(2, r2t));
        temp = mul(temp, mul(t[x], RM));
        r = add(r, temp);
        r = add(r, mul(C2(t[x]), C2(p)));
        rx -= 1;
        r = add(r, mul(C2(t[x]), mul(rx, p)));

        if (countIfEqual) {
            r = add(r, mul(mul(t[x], sumt), mul(rx, p)));
            r = add(r, mul(C2(t[x]), C2(rx)));
        }

        t[x]++;
        R2 = add(r2, sqr(rx));
        RT = add(rt, mul(rx, t[x]));
        R2T = add(r2t, mul(sqr(rx), t[x]));
    }

    return r;
}

class Solution {
public:
    int subsequencesWithMiddleMode(vector<int>& nums) {
        unordered_map<int, int> mp;
        int m = 0;
        for (int& x : nums) {
            if (!mp.count(x)) mp[x] = m++;
            x = mp[x];
        }

        vector<int> c(m);
        for (int x : nums) ++c[x];

        int r = count(nums, c, true);
        reverse(nums.begin(), nums.end());
        r = add(r, count(nums, c, false));
        return r;
    }
};
