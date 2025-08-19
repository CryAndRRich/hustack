#include <vector>
#include <stack>
#include <unordered_set>
using namespace std;

class Solution {
public:
    int minRunesToAdd(int n, vector<int>& crystals, vector<int>& flowFrom, vector<int>& flowTo) {
        vector<vector<int>> adj(n), transpose(n);
        for (int i = 0; i < (int)flowFrom.size(); i++) {
            adj[flowFrom[i]].push_back(flowTo[i]);
            transpose[flowTo[i]].push_back(flowFrom[i]);
        }
        
        vector<bool> visited(n, false);
        stack<int> st;
        for (int i = 0; i < n; i++) if (!visited[i]) dfs(i, adj, visited, st);
        
        fill(visited.begin(), visited.end(), false);
        vector<vector<int>> sccs;
        while (!st.empty()) {
            int node = st.top(); st.pop();
            if (!visited[node]) {
                vector<int> scc;
                dfsTranspose(node, transpose, visited, scc);
                sccs.push_back(move(scc));
            }
        }
        
        vector<int> sccMap(n);
        for (int i = 0; i < (int)sccs.size(); i++) {
            for (int node : sccs[i]) sccMap[node] = i;
        }
        
        vector<int> indegree((int)sccs.size(), 0);
        unordered_set<long long> edgeSet;
        for (int u = 0; u < n; u++) {
            for (int v : adj[u]) {
                if (sccMap[u] != sccMap[v]) {
                    long long code = ((long long)sccMap[u] << 32) | sccMap[v];
                    if (edgeSet.find(code) == edgeSet.end()) {
                        edgeSet.insert(code);
                        indegree[sccMap[v]]++;
                    }
                }
            }
        }
        
        unordered_set<int> crystalScc;
        for (int c : crystals) crystalScc.insert(sccMap[c]);
        
        int count = 0;
        for (int i = 0; i < (int)sccs.size(); i++) {
            if (indegree[i] == 0 && crystalScc.count(i) == 0) count++;
        }
        return count;
    }
    
private:
    void dfs(int u, vector<vector<int>>& adj, vector<bool>& visited, stack<int>& st) {
        visited[u] = true;
        for (int v : adj[u]) {
            if (!visited[v]) dfs(v, adj, visited, st);
        }
        st.push(u);
    }
    
    void dfsTranspose(int u, vector<vector<int>>& transpose, vector<bool>& visited, vector<int>& scc) {
        visited[u] = true;
        scc.push_back(u);
        for (int v : transpose[u]) {
            if (!visited[v]) dfsTranspose(v, transpose, visited, scc);
        }
    }
};
