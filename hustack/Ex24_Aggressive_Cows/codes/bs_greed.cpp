#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

bool greedy(const vector<int>& pos, int C, int dist) {
    int count = 1;
    int last_pos = pos[0];

    for (int i = 1; i < pos.size(); ++i) {
        if (pos[i] - last_pos >= dist) {
            count++;
            last_pos = pos[i];
        }
        if (count >= C)
            return true;
    }
    return false;
}

int binary_search(const vector<int>& pos, int C) {
    int low = 1;
    int high = pos.back() - pos[0];
    int ans = 0;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (greedy(pos, C, mid)) {
            ans = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return ans;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    cin >> T;

    while (T--) {
        int N, C;
        cin >> N >> C;

        vector<int> pos(N);
        for (int i = 0; i < N; ++i) {
            cin >> pos[i];
        }

        sort(pos.begin(), pos.end());

        int res = binary_search(pos, C);
        cout << res;
    }

    return 0;
}
