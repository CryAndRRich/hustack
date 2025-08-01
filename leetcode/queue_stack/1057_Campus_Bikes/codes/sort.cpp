#include <vector>
#include <tuple>
#include <algorithm>
using namespace std;

class Solution {
public:
    vector<int> assignBikes(vector<vector<int>>& workers, vector<vector<int>>& bikes) {
        vector<tuple<int, int, int>> pairs;

        for (int i = 0; i < workers.size(); ++i) {
            for (int j = 0; j < bikes.size(); ++j) {
                int dist = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1]);
                pairs.emplace_back(dist, i, j);
            }
        }

        sort(pairs.begin(), pairs.end());
        vector<int> res(workers.size(), -1);
        vector<bool> used(bikes.size(), false);

        for (auto [dist, i, j] : pairs) {
            if (res[i] == -1 && !used[j]) {
                res[i] = j;
                used[j] = true;
            }
        }

        return res;
    }
};
