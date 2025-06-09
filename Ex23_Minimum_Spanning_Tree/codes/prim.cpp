#include <iostream>
#include <vector>
#include <queue>
#include <utility> 
#include <cstring> 

using namespace std;

const int MAXN = 100005;
vector<pair<int, int>> graph[MAXN];
bool visited[MAXN];

int prim(int start) {
    memset(visited, false, sizeof(visited));
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
    pq.push({0, start});
    int total = 0;

    while (!pq.empty()) {
        auto [w, u] = pq.top(); pq.pop();
        if (visited[u]) continue;
        visited[u] = true;
        total += w;
        for (auto &[v, wt] : graph[u]) {
            if (!visited[v]) {
                pq.push({wt, v});
            }
        }
    }

    return total;
}

int main() {
    int n, m; cin >> n >> m;
    for (int i = 0; i < m; ++i) {
        int u, v, w; cin >> u >> v >> w;
        graph[u].emplace_back(v, w);
        graph[v].emplace_back(u, w);
    }

    int res = prim(1);
    cout << res;
    return 0;
}
