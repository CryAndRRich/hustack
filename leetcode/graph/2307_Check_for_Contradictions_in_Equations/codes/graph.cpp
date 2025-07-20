#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
using namespace std;

class Solution {
public:
    bool checkContradictions(vector<vector<string>>& equations, vector<double>& values) {
        unordered_map<string, string> parent;
        unordered_map<string, double> weight;

        function<string(string)> find = [&](string x) -> string {
            if (parent[x] != x) {
                string orig = parent[x];
                parent[x] = find(orig);
                weight[x] *= weight[orig];
            }
            return parent[x];
        };

        for (int i = 0; i < equations.size(); ++i) {
            string a = equations[i][0], b = equations[i][1];
            double val = values[i];

            if (!parent.count(a)) {
                parent[a] = a;
                weight[a] = 1.0;
            }
            if (!parent.count(b)) {
                parent[b] = b;
                weight[b] = 1.0;
            }

            string rootA = find(a);
            string rootB = find(b);

            if (rootA == rootB) {
                double expected = weight[a] / weight[b];
                if (abs(expected - val) > 1e-5) return true;
            } else {
                parent[rootA] = rootB;
                weight[rootA] = val * weight[b] / weight[a];
            }
        }

        return false;
    }
};
