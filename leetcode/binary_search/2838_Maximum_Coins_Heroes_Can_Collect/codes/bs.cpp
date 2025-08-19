#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    vector<long long> maximumCoins(vector<int>& heroes, vector<int>& monsters, vector<int>& coins) {
        int m = monsters.size();
        vector<pair<int, int>> paired(m);
        for (int i = 0; i < m; ++i)
            paired[i] = {monsters[i], coins[i]};

        sort(paired.begin(), paired.end());

        vector<int> monsters_sorted(m);
        vector<long long> prefix(m + 1, 0);
        for (int i = 0; i < m; ++i) {
            monsters_sorted[i] = paired[i].first;
            prefix[i + 1] = prefix[i] + paired[i].second;
        }

        vector<long long> result;
        for (int h : heroes) {
            int idx = upper_bound(monsters_sorted.begin(), monsters_sorted.end(), h) - monsters_sorted.begin();
            result.push_back(prefix[idx]);
        }
        return result;
    }
};
