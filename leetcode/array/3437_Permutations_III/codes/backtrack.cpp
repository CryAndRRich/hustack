#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<vector<int>> permute(int n) {
        vector<vector<int>> res;
        vector<int> path;
        vector<bool> used(n + 1, false);
        function<void()> backtrack = [&]() {
            if (path.size() == n) {
                res.push_back(path);
                return;
            }
            for (int num = 1; num <= n; num++) {
                if (!used[num]) {
                    if (path.empty() || (path.back() % 2 != num % 2)) {
                        used[num] = true;
                        path.push_back(num);
                        backtrack();
                        path.pop_back();
                        used[num] = false;
                    }
                }
            }
        };
        backtrack();
        return res;
    }
};
