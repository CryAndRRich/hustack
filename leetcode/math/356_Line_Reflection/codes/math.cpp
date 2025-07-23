using namespace std;

class Solution {
public:
    bool isReflected(vector<vector<int>>& points) {
        unordered_set<string> pointSet;
        int minX = INT_MAX, maxX = INT_MIN;

        for (auto& p : points) {
            minX = min(minX, p[0]);
            maxX = max(maxX, p[0]);
            pointSet.insert(to_string(p[0]) + "#" + to_string(p[1]));
        }

        int reflectX = minX + maxX;

        for (auto& p : points) {
            int reflectedX = reflectX - p[0];
            string key = to_string(reflectedX) + "#" + to_string(p[1]);
            if (pointSet.find(key) == pointSet.end())
                return false;
        }

        return true;
    }
};
