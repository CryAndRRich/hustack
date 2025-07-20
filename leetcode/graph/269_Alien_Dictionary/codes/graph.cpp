#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
using namespace std;

class Solution {
public:
    string alienOrder(vector<string>& words) {
        unordered_map<char, unordered_set<char>> graph;
        unordered_map<char, int> in_degree;

        for (string& word : words)
            for (char c : word)
                in_degree[c] = 0;

        for (int i = 0; i < words.size() - 1; ++i) {
            string& w1 = words[i], &w2 = words[i + 1];
            if (w1.size() > w2.size() && w1.find(w2) == 0)
                return "";
            for (int j = 0; j < min(w1.size(), w2.size()); ++j) {
                char c1 = w1[j], c2 = w2[j];
                if (c1 != c2) {
                    if (!graph[c1].count(c2)) {
                        graph[c1].insert(c2);
                        in_degree[c2]++;
                    }
                    break;
                }
            }
        }

        queue<char> q;
        for (auto& [c, deg] : in_degree)
            if (deg == 0) q.push(c);

        string res;
        while (!q.empty()) {
            char c = q.front(); q.pop();
            res += c;
            for (char nei : graph[c]) {
                if (--in_degree[nei] == 0)
                    q.push(nei);
            }
        }

        return res.size() == in_degree.size() ? res : "";
    }
};
