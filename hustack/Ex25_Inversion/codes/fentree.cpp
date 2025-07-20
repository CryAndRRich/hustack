#include <iostream>
#include <vector>
using namespace std;

const int MAXV = 1e6 + 5;
const int MOD = 1e9 + 7;

vector<int> bit(MAXV);

int query(int x) {
    int sum = 0;
    while (x > 0) {
        sum = (sum + bit[x]) % MOD;
        x -= x & -x;
    }
    return sum;
}

void update(int x) {
    while (x < MAXV) {
        bit[x] = (bit[x] + 1) % MOD;
        x += x & -x;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    vector<int> a(n);

    for (int &x : a) cin >> x;

    long long res = 0;

    for (int i = n - 1; i >= 0; --i) {
        res = (res + query(a[i] - 1)) % MOD;
        update(a[i]);
    }

    cout << res;
    return 0;
}
