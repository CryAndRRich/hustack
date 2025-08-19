#include <bits/stdc++.h>
using namespace std;

class cmp {
public:
    bool operator() (const pair<int, int>& a, const pair<int, int>& b) const {
        if (a.first != b.first) return a.first < b.first;
        return a.second > b.second;
    }
};

class StatisticsTracker {
private:
    long long sum = 0;
    deque<int> nums;
    multiset<int, greater<>> upper;
    multiset<int> lower;
    unordered_map<int, int> numFreq;
    set<pair<int, int>, cmp> freqNum;

    void balance() {
        if (upper.size() == lower.size() - 1) {
            upper.insert(*prev(lower.end()));
            lower.erase(prev(lower.end()));
        }
        if (lower.size() == upper.size() - 2) {
            lower.insert(*prev(upper.end()));
            upper.erase(prev(upper.end()));
        }
    }

public:
    void addNumber(int number) {
        sum += number;
        nums.push_back(number);

        int prevFreq = numFreq[number]++;
        int currFreq = prevFreq + 1;
        if (prevFreq) freqNum.erase({prevFreq, number});
        freqNum.insert({currFreq, number});

        upper.insert(number);
        lower.insert(*prev(upper.end()));
        upper.erase(prev(upper.end()));
        if (upper.size() < lower.size()) {
            upper.insert(*prev(lower.end()));
            lower.erase(prev(lower.end()));
        }
    }

    void removeFirstAddedNumber() {
        int number = nums.front(); nums.pop_front();
        sum -= number;

        int prevFreq = numFreq[number]--;
        int currFreq = prevFreq - 1;
        freqNum.erase({prevFreq, number});
        if (currFreq) freqNum.insert({currFreq, number});
        else numFreq.erase(number);

        if (number >= *prev(upper.end())) upper.erase(upper.find(number));
        else lower.erase(lower.find(number));
        balance();
    }

    int getMean() {
        return sum / nums.size();
    }

    int getMedian() {
        return *prev(upper.end());
    }

    int getMode() {
        return prev(freqNum.end())->second;
    }
};

/**
 * Your StatisticsTracker object will be instantiated and called as such:
 * StatisticsTracker* obj = new StatisticsTracker();
 * obj->addNumber(number);
 * obj->removeFirstAddedNumber();
 * int param_3 = obj->getMean();
 * int param_4 = obj->getMedian();
 * int param_5 = obj->getMode();
 */