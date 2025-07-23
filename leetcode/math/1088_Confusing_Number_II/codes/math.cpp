#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    int confusingNumberII(int n) {
        vector<int> digits = {0, 1, 6, 8, 9};
        unordered_map<char, char> mapping = {
            {'0', '0'}, {'1', '1'}, {'6', '9'}, {'8', '8'}, {'9', '6'}
        };
        string s = to_string(n);
        int half = s.size() / 2;
        bool odd = s.size() % 2;
        int ans = 2;

        for (int l = 2; l < s.size(); l++) {
            int tmp = pow(5, l - 1);
            int sub = pow(5, l / 2 - 1) * (l % 2 ? 3 : 1);
            ans += 4 * (tmp - sub);
        }

        ans += calc(s, 0, digits, mapping, half, odd);

        int first_half_num = 0;
        for (int i = half - 1; i >= 0; i--) {
            if (!mapping.count(s[i])) return ans;
            first_half_num = first_half_num * 10 + (mapping[s[i]] - '0');
        }

        if (odd) {
            if (s[half] == '0' || s[half] == '1' || s[half] == '8') {
                if (stoi(s.substr(half + 1)) >= first_half_num) ans--;
            }
        } else {
            if (stoi(s.substr(half)) >= first_half_num) ans--;
        }
        return ans;
    }

    int calc(const string &s, int idx, const vector<int>& digits, unordered_map<char, char> &mapping, int half, bool odd) {
        int num = s[idx] - '0';
        int index = (num > 8) ? 4 : (num > 6) ? 3 : (num > 1) ? 2 : (num == 1) ? 1 : 0;

        if (idx == s.size() - 1) return index + (digits[index] == num);

        int smaller = (idx == 0 ? index - 1 : index);
        int res = smaller * pow(5, s.size() - idx - 1);

        if (idx < half)
            res -= smaller * pow(5, half - idx - 1) * (odd ? 3 : 1);
        if (idx == half && odd)
            res -= (num > 8 ? 3 : (num > 1 ? 2 : (num > 0 ? 1 : 0)));
        if (digits[index] == num)
            res += calc(s, idx + 1, digits, mapping, half, odd);
        return res;
    }
};
