#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

using namespace std;

class Solution {
public:
    string rearrangeString(string s, int k) {
        if (k == 0) return s;

        unordered_map<char, int> freq;
        for (char c : s) freq[c]++;

        int mf = 0;
        vector<char> mfch;
        for (auto& [ch, c] : freq) {
            if (c > mf) {
                mf = c;
                mfch = {ch};
            } else if (c == mf) {
                mfch.push_back(ch);
            }
        }

        string head = "";
        for (char ch : mfch) head += ch;
        vector<string> res(mf, head);

        if (k * (mf - 1) + head.length() > s.length()) return "";

        vector<pair<int, char>> A;
        for (auto& [ch, c] : freq) {
            if (c != mf) A.emplace_back(c, ch);
        }
        sort(A.rbegin(), A.rend());

        int j = 0;
        for (auto& [count, ch] : A) {
            for (int i = 0; i < count; ++i) {
                res[j] += ch;
                j = (j + 1) % (mf - 1);
            }
        }

        string ans = "";
        for (string& part : res) ans += part;
        return ans;
    }
};
