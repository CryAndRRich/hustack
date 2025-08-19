#include <vector>
#include <numeric>
#include <algorithm>

using namespace std;

class Solution {
public:
    long long maxGcdSum(vector<int>& nums, int k) {
        int n = nums.size();
        vector<long long> prefix(n + 1, 0);
        for (int i = 0; i < n; ++i)
            prefix[i + 1] = prefix[i] + nums[i];

        long long best = 0;
        vector<pair<int, int>> gcds; // (index, gcd)

        for (int i = 0; i < n; ++i) {
            vector<pair<int, int>> new_gcds;
            gcds.push_back({i, nums[i]});
            for (auto [j, g] : gcds) {
                int ng = gcd(g, nums[i]);
                if (new_gcds.empty() || new_gcds.back().second != ng)
                    new_gcds.push_back({j, ng});
            }
            gcds = new_gcds;

            for (auto [j, g] : gcds) {
                if (i - j + 1 < k) break;
                long long sum = prefix[i + 1] - prefix[j];
                best = max(best, sum * g);
            }
        }

        return best;
    }
};
