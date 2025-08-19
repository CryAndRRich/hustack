#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

class Solution {
public:
    bool check(vector<int>& timeReq, long long maxTime, int splitTime) {
        int currentWBC = 1;
        long long curTime = 0;
        for (int i = 0; i < timeReq.size(); i++) {
            if (currentWBC == 0) return false;
            while (curTime + splitTime + timeReq[i] <= maxTime && currentWBC < timeReq.size() - i) {
                currentWBC <<= 1;
                curTime += splitTime;
            }
            currentWBC--;
        }
        return true;
    }

    long long minEliminationTime(vector<int>& timeReq, int splitTime) {
        sort(timeReq.begin(), timeReq.end(), greater<int>());
        long long left = timeReq[0], right = 20LL * splitTime + timeReq[0];
        while (right > left) {
            long long answer = (left + right) / 2;
            if (check(timeReq, answer, splitTime)) right = answer;
            else left = answer + 1;
        }
        return left;
    }
};
