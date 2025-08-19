#include <vector>
#include <stack>
#include <cstring>
using namespace std;

class Solution {
public:
    int n;
    vector<vector<int>> graph;
    vector<int> index, lowlink, scc_id, indegree;
    vector<bool> on_stack;
    stack<int> stk;
    int idx = 1, scc_count = 0;

    void tarjan(int u) {
        index[u] = idx;
        lowlink[u] = idx;
        idx++;
        stk.push(u);
        on_stack[u] = true;

        for (int v : graph[u]) {
            if (index[v] == 0) {
                tarjan(v);
                lowlink[u] = min(lowlink[u], lowlink[v]);
            }
            else if (on_stack[v]) {
                lowlink[u] = min(lowlink[u], index[v]);
            }
        }

        if (lowlink[u] == index[u]) {
            while (true) {
                int node = stk.top();
                stk.pop();
                on_stack[node] = false;
                scc_id[node] = scc_count;
                if (node == u) break;
            }
            scc_count++;
        }
    }

    int minRunesToAdd(int n_, vector<int>& crystals, vector<int>& flowFrom, vector<int>& flowTo) {
        n = n_;
        graph.assign(n, vector<int>());
        for (int i = 0; i < (int)flowFrom.size(); i++) {
            graph[flowFrom[i]].push_back(flowTo[i]);
        }

        index.assign(n, 0);
        lowlink.assign(n, 0);
        scc_id.assign(n, -1);
        on_stack.assign(n, false);
        idx = 1;
        scc_count = 0;

        for (int i = 0; i < n; i++) {
            if (index[i] == 0) {
                tarjan(i);
            }
        }

        indegree.assign(scc_count, 0);
        for (int u = 0; u < n; u++) {
            for (int v : graph[u]) {
                if (scc_id[u] != scc_id[v]) {
                    indegree[scc_id[v]]++;
                }
            }
        }

        vector<bool> crystal_scc(scc_count, false);
        for (int c : crystals) {
            crystal_scc[scc_id[c]] = true;
        }

        int result = 0;
        for (int i = 0; i < scc_count; i++) {
            if (indegree[i] == 0 && !crystal_scc[i]) result++;
        }

        return result;
    }
};
