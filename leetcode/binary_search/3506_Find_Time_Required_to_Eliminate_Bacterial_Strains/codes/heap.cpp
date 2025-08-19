#include <iostream>
#include <vector>
#include <queue>
using namespace std;

class Solution {
public:
    long long minEliminationTime(vector<int>& timeReq, int splitTime) {
        priority_queue<long long, vector<long long>, greater<long long>> pq;

        for (int t : timeReq)
            pq.push(t);

        while (pq.size() > 1) {
            long long t1 = pq.top(); pq.pop();
            long long t2 = pq.top(); pq.pop();

            long long merged = t2 + splitTime;
            pq.push(merged);
        }

        return pq.top();
    }
};
