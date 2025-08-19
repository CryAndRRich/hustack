#include <vector>
#include <algorithm>
#include <cstring>
#include <unordered_map>
using namespace std;

class Solution {
public:
    vector<int> parent, ancestor;
    vector<bool> visited;
    vector<vector<pair<int,int>>> tree;
    vector<vector<int>> freq;
    vector<vector<pair<int,int>>> queries;
    vector<int> ans;

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    void unite(int x, int y) {
        parent[find(x)] = find(y);
    }

    void dfs(int u, int p) {
        visited[u] = true;
        ancestor[u] = u;
        for (auto [v, w] : tree[u]) {
            if (v == p) continue;
            freq[v] = freq[u];
            freq[v][w]++;
            dfs(v, u);
            unite(v, u);
            ancestor[find(u)] = u;
        }
        for (auto [v, idx] : queries[u]) {
            if (visited[v]) {
                int lca = ancestor[find(v)];
                vector<int> count(27, 0);
                for (int k = 1; k <= 26; k++)
                    count[k] = freq[u][k] + freq[v][k] - 2 * freq[lca][k];
                int total = 0, mx = 0;
                for (int k = 1; k <= 26; k++) {
                    total += count[k];
                    mx = max(mx, count[k]);
                }
                ans[idx] = total - mx;
            }
        }
    }

    vector<int> minOperationsQueries(int n, vector<vector<int>>& edges, vector<vector<int>>& q) {
        tree.assign(n, {});
        for (auto &e : edges) {
            tree[e[0]].push_back({e[1], e[2]});
            tree[e[1]].push_back({e[0], e[2]});
        }
        parent.resize(n);
        ancestor.resize(n);
        visited.assign(n, false);
        freq.assign(n, vector<int>(27, 0));
        queries.assign(n, {});
        ans.resize(q.size());

        for (int i = 0; i < n; i++) parent[i] = i;
        for (int i = 0; i < (int)q.size(); i++) {
            queries[q[i][0]].push_back({q[i][1], i});
            queries[q[i][1]].push_back({q[i][0], i});
        }
        dfs(0, -1);
        return ans;
    }
};
