#include <bits/stdc++.h>
using namespace std;

class HitCounter {
private:
    int total;
    queue<pair<int, int>> hits;
public:
    HitCounter() {
        total = 0;
    }
    void hit(int timestamp) {
        if (hits.empty() || hits.back().first != timestamp) {
            hits.push({timestamp, 1});
        } else {
            auto last = hits.back();
            hits.pop();
            last.second++;
            hits.push(last);
        }
        total++;
    }
    int getHits(int timestamp) {
        while (!hits.empty() && timestamp - hits.front().first >= 300) {
            total -= hits.front().second;
            hits.pop();
        }
        return total;
    }
};
