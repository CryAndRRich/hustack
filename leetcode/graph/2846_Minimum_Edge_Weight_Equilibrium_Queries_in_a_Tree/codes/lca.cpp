#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    vector<int> minOperationsQueries(int n, vector<vector<int>>& edges, vector<vector<int>>& queries) {
        vector<vector<pair<int,int>>> tree(n);
        for (auto &e : edges) {
            tree[e[0]].push_back({e[1], e[2]});
            tree[e[1]].push_back({e[0], e[2]});
        }
        int m = log2(n) + 2;
        vector<vector<int>> lift(n, vector<int>(m, -1));
        vector<vector<int>> freq(n, vector<int>(27, 0));
        vector<int> depth(n, -1);
        stack<tuple<int,int,int>> st;
        st.push({0, -1, 0});
        while (!st.empty()) {
            auto [u, p, d] = st.top(); st.pop();
            depth[u] = d;
            for (auto &[v, w] : tree[u]) {
                if (v != p) {
                    lift[v][0] = u;
                    freq[v] = freq[u];
                    freq[v][w]++;
                    for (int j = 1; j < m; j++) {
                        if (lift[v][j-1] == -1) break;
                        lift[v][j] = lift[lift[v][j-1]][j-1];
                    }
                    st.push({v, u, d+1});
                }
            }
        }
        auto lca = [&](int u, int v) {
            if (depth[u] > depth[v]) swap(u, v);
            for (int i = 0; i < m; i++) {
                if ((depth[v] - depth[u]) >> i & 1) v = lift[v][i];
            }
            if (u == v) return u;
            for (int i = m - 1; i >= 0; i--) {
                if (lift[u][i] != lift[v][i]) {
                    u = lift[u][i];
                    v = lift[v][i];
                }
            }
            return lift[u][0];
        };
        vector<int> ans;
        for (auto &q : queries) {
            int u = q[0], v = q[1];
            int k = lca(u, v);
            vector<int> count(27, 0);
            for (int w = 1; w <= 26; w++)
                count[w] = freq[u][w] + freq[v][w] - 2 * freq[k][w];
            ans.push_back(accumulate(count.begin(), count.end(), 0) - *max_element(count.begin(), count.end()));
        }
        return ans;
    }
};
