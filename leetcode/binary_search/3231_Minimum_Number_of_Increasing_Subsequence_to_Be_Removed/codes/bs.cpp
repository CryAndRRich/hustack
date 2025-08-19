#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    int minOperations(vector<int>& nums) {
        vector<int> tails;
        for (int x : nums) {
            int neg = -x;
            auto it = upper_bound(tails.begin(), tails.end(), neg);
            if (it == tails.end())
                tails.push_back(neg);
            else
                *it = neg;
        }
        return tails.size();
    }
};
