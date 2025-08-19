#include <vector>
#include <tuple>
#include <limits>
#include <unordered_map>
#include <algorithm>
#include <array>
#include <cmath>
using namespace std;

class Solution {
public:
    vector<int> beautifulPair(vector<int>& nums1, vector<int>& nums2) {
        unordered_map<long long, int> seen;
        int n = nums1.size();
        pair<int, int> res = {numeric_limits<int>::max(), numeric_limits<int>::max()};

        auto hash = [](int x, int y) { return ((long long)x << 32) | (unsigned int)y; };

        for (int i = 0; i < n; ++i) {
            long long h = hash(nums1[i], nums2[i]);
            if (seen.count(h)) {
                pair<int, int> cur = {seen[h], i};
                if (cur < res) res = cur;
            } else {
                seen[h] = i;
            }
        }
        if (res.first < numeric_limits<int>::max()) {
            return {res.first, res.second};
        }

        vector<array<int, 3>> points;
        for (int i = 0; i < n; ++i) points.push_back({nums1[i], nums2[i], i});
        sort(points.begin(), points.end());
        auto [mind, resi, resj] = closest(points);
        if (resi > resj) swap(resi, resj);
        return {resi, resj};
    }

private:
    static tuple<int, int, int> closest(const vector<array<int, 3>>& points) {
        int n = points.size();
        if (n < 2) return {numeric_limits<int>::max(), numeric_limits<int>::max(), numeric_limits<int>::max()};
        int mid = n / 2;
        const auto& midP = points[mid];

        auto [dl, pi, pj] = closest(vector<array<int, 3>>(points.begin(), points.begin() + mid));
        auto [dr, qi, qj] = closest(vector<array<int, 3>>(points.begin() + mid, points.end()));
        tuple<int, int, int> dres = min(make_tuple(dl, pi, pj), make_tuple(dr, qi, qj));
        int d = get<0>(dres);

        vector<array<int, 3>> strip;
        for (int i = 0; i < n; ++i)
            if (abs(points[i][0] - midP[0]) <= d) strip.push_back(points[i]);

        auto [ds, si, sj] = stripclosest(strip, d);
        return min(dres, make_tuple(ds, min(si, sj), max(si, sj)));
    }

    static tuple<int, int, int> stripclosest(vector<array<int, 3>>& strip, int d) {
        int min_d = d;
        int si = numeric_limits<int>::max(), sj = numeric_limits<int>::max();
        sort(strip.begin(), strip.end(), [](auto &a, auto &b) {
            return a[1] != b[1] ? a[1] < b[1] : a[2] < b[2];
        });
        int m = strip.size();
        for (int i = 0; i < m; ++i) {
            for (int j = i+1; j < m; ++j) {
                if (strip[j][1] - strip[i][1] > min_d) break;
                int currd = abs(strip[j][0] - strip[i][0]) + abs(strip[j][1] - strip[i][1]);
                int curri = min(strip[i][2], strip[j][2]);
                int currj = max(strip[i][2], strip[j][2]);
                if (make_tuple(currd, curri, currj) < make_tuple(min_d, si, sj))
                    tie(min_d, si, sj) = make_tuple(currd, curri, currj);
            }
        }
        return {min_d, si, sj};
    }
};
