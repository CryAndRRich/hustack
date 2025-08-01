#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>

using namespace std;

class Solution {
public:
    string rearrangeString(string s, int k) {
        if (k == 0) return s;

        unordered_map<char, int> freq;
        for (char c : s) freq[c]++;

        priority_queue<pair<int, char>> maxHeap;
        for (auto &[ch, count] : freq)
            maxHeap.push({count, ch});

        queue<tuple<int, char, int>> waitQueue; 
        string result;

        while (!maxHeap.empty() || !waitQueue.empty()) {
            if (!waitQueue.empty() && get<2>(waitQueue.front()) <= result.size()) {
                auto [cnt, ch, _] = waitQueue.front(); waitQueue.pop();
                maxHeap.push({cnt, ch});
            }

            if (maxHeap.empty())
                return result.size() == s.size() ? result : "";

            auto [cnt, ch] = maxHeap.top(); maxHeap.pop();
            result += ch;

            if (--cnt > 0)
                waitQueue.push({cnt, ch, (int)result.size() + k - 1});
        }

        return result;
    }
};
