#include <vector>
#include <queue>
#include <unordered_map>
using namespace std;

class Solution {
public:
    long long maxProfit(vector<int>& workers, vector<vector<int>>& tasks) {
        long long res = 0;
        int additionalWorker = 1;
        priority_queue<pair<int,int>> pq; 
        for (auto& t : tasks) {
            pq.push({t[1], t[0]});
        }
        unordered_map<int,int> wrks;
        for (int w : workers) {
            wrks[w]++;
        }
        while (!pq.empty()) {
            auto [profit, skill] = pq.top();
            pq.pop();
            if (wrks[skill] > 0 || additionalWorker > 0) {
                res += profit;
                if (wrks[skill] > 0) {
                    wrks[skill]--;
                } else {
                    additionalWorker = 0;
                }
            }
        }
        return res;
    }
};
