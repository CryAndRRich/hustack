#include <bits/stdc++.h>
using namespace std;

struct edge { 
    int to, rev; 
    int cap; 
    long long cost; 
};

struct mcmf {
    int n;
    vector<vector<edge>> g;
    vector<long long> dist, potential;
    vector<int> pv_v, pv_e;

    mcmf(int n): n(n), g(n), dist(n), potential(n), pv_v(n), pv_e(n) {}

    void addEdge(int u, int v, int cap, long long cost) {
        g[u].push_back({v, (int)g[v].size(), cap, cost});
        g[v].push_back({u, (int)g[u].size()-1, 0, -cost});
    }

    pair<int,long long> minCostMaxFlow(int s, int t, int maxf) {
        int flow = 0;
        long long cost = 0;
        fill(potential.begin(), potential.end(), 0);
        while (flow < maxf) {
            priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
            fill(dist.begin(), dist.end(), INT_MAX);
            dist[s] = 0;
            pq.push({0, s});
            while (!pq.empty()) {
                auto [d, u] = pq.top(); 
                pq.pop();
                if (d > dist[u]) continue;
                for (int i = 0; i < (int)g[u].size(); i++) {
                    edge &e = g[u][i];
                    if (e.cap <= 0) continue;
                    long long nd = d + e.cost + potential[u] - potential[e.to];
                    if (nd < dist[e.to]) {
                        dist[e.to] = nd;
                        pv_v[e.to] = u;
                        pv_e[e.to] = i;
                        pq.push({nd, e.to});
                    }
                }
            }
            if (dist[t] == INT_MAX) break;
            for (int v = 0; v < n; v++)
                if (dist[v] < INT_MAX) potential[v] += dist[v];
            int addf = maxf - flow;
            int v = t;
            while (v != s) {
                addf = min(addf, g[pv_v[v]][pv_e[v]].cap);
                v = pv_v[v];
            }
            flow += addf;
            cost += (long long)addf * potential[t];
            v = t;
            while (v != s) {
                edge &e = g[pv_v[v]][pv_e[v]];
                e.cap -= addf;
                g[v][e.rev].cap += addf;
                v = pv_v[v];
            }
        }
        return {flow, cost};
    }
};

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    cin >> n >> m;
    mcmf mcmf(n + 1);
    int u, v;
    long long c;
    for(int i = 0; i < m; i++){
        cin >> u >> v >> c;
        mcmf.addEdge(u, v, 1, c);
    }
    auto res = mcmf.minCostMaxFlow(1, n, 2);
    if(res.first < 2) {
        cout << "NOT_FEASIBLE";
    } else {
        cout << res.second;
    }
    return 0;
}
