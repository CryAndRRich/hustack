#include <vector>
#include <queue>
#include <climits>
using namespace std;

class Solution {
public:
    vector<vector<int>> graph;

    vector<int> closestNode(int n, vector<vector<int>>& edges, vector<vector<int>>& query) {
        graph = vector<vector<int>>(n);
        for (auto& e : edges) {
            graph[e[0]].push_back(e[1]);
            graph[e[1]].push_back(e[0]);
        }

        vector<int> res;
        for (auto& q : query) {
            int start = q[0], end = q[1], node = q[2];
            vector<int> path;
            dfsPath(start, end, -1, path);

            vector<int> dist = bfsDist(n, node);
            int minNode = path[0];
            for (int p : path) {
                if (dist[p] < dist[minNode]) {
                    minNode = p;
                }
            }
            res.push_back(minNode);
        }

        return res;
    }

    void dfsPath(int u, int target, int parent, vector<int>& path) {
        path.push_back(u);
        if (u == target) return;
        for (int v : graph[u]) {
            if (v != parent) {
                int prevSize = path.size();
                dfsPath(v, target, u, path);
                if (path.back() == target) return;
                path.resize(prevSize);
            }
        }
    }

    vector<int> bfsDist(int n, int src) {
        vector<int> dist(n, -1);
        queue<int> q;
        q.push(src);
        dist[src] = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : graph[u]) {
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        return dist;
    }
};
