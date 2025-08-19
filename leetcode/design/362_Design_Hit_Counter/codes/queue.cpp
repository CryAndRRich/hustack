#include <bits/stdc++.h>
using namespace std;

class HitCounter {
private:
    queue<int> hits;
public:
    HitCounter() {}
    void hit(int timestamp) {
        hits.push(timestamp);
    }
    int getHits(int timestamp) {
        while (!hits.empty() && timestamp - hits.front() >= 300) {
            hits.pop();
        }
        return hits.size();
    }
};
