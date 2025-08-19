#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    long long maxArea(int height, vector<int>& positions, string directions) {
        map<int, vector<int>> have;
        int n = positions.size();
        long long s = 0;
        int a = 0;
        for (int i = 0; i < n; i++) {
            s += positions[i];
            if (directions[i] == 'U') {
                a++;
                have[height - positions[i]].push_back(i);
                have[height - positions[i] + height].push_back(i);
            } else {
                a--;
                have[positions[i]].push_back(i);
                have[positions[i] + height].push_back(i);
            }
        }
        long long r = s;
        int now = 0;
        for (auto &p : have) {
            long long t = p.first - now;
            s += t * a;
            now = p.first;
            r = max(r, s);
            for (int i : p.second) {
                if (directions[i] == 'U') {
                    directions[i] = 'D';
                    a -= 2;
                } else {
                    directions[i] = 'U';
                    a += 2;
                }
            }
        }
        return r;
    }
};
