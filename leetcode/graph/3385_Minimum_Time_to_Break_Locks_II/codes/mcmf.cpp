#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int to, cap, cost, rev;
};

struct MinCostMaxFlow {
    int N;
    vector<vector<Edge>> adj;
    MinCostMaxFlow(int n) : N(n), adj(n) {}

    void add_edge(int u, int v, int cap, int cost) {
        adj[u].push_back({v, cap, cost, (int)adj[v].size()});
        adj[v].push_back({u, 0, -cost, (int)adj[u].size() - 1});
    }

    int flow(int s, int t, int maxf) {
        const int INF = 1e9;
        vector<int> h(N, 0), dist(N), prevv(N), preve(N);
        int res = 0, flow = 0;

        while (flow < maxf) {
            fill(dist.begin(), dist.end(), INF);
            dist[s] = 0;
            vector<bool> inqueue(N, false);
            queue<int> q;
            q.push(s);
            inqueue[s] = true;

            while (!q.empty()) {
                int u = q.front(); q.pop();
                inqueue[u] = false;
                for (int i = 0; i < (int)adj[u].size(); i++) {
                    Edge &e = adj[u][i];
                    if (e.cap > 0 && dist[e.to] > dist[u] + e.cost + h[u] - h[e.to]) {
                        dist[e.to] = dist[u] + e.cost + h[u] - h[e.to];
                        prevv[e.to] = u;
                        preve[e.to] = i;
                        if (!inqueue[e.to]) {
                            q.push(e.to);
                            inqueue[e.to] = true;
                        }
                    }
                }
            }

            if (dist[t] == INF) return res;
            for (int i = 0; i < N; i++) h[i] += dist[i];

            int d = maxf - flow;
            for (int v = t; v != s; v = prevv[v])
                d = min(d, adj[prevv[v]][preve[v]].cap);

            flow += d;
            res += d * h[t];
            for (int v = t; v != s; v = prevv[v]) {
                Edge &e = adj[prevv[v]][preve[v]];
                e.cap -= d;
                adj[v][e.rev].cap += d;
            }
        }
        return res;
    }
};

class Solution {
public:
    int findMinimumTime(vector<int>& strength) {
        int n = strength.size();
        int source = 2 * n;
        int sink = 2 * n + 1;
        MinCostMaxFlow mcmf(2 * n + 2);

        for (int i = 0; i < n; i++) {
            mcmf.add_edge(source, i, 1, 0);
            mcmf.add_edge(n + i, sink, 1, 0);
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int cost = (strength[j] + i) / (i + 1);
                mcmf.add_edge(i, n + j, 1, cost);
            }
        }

        return mcmf.flow(source, sink, n);
    }
};
