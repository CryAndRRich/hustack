#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
using namespace std;

int n, m;
vector<set<int>> graph;
int ans = 0;
vector<int> best;

void bron_kerbosch(set<int> R, set<int> P, set<int> X) {
    if (P.empty() && X.empty()) {
        if ((int)R.size() > ans) {
            ans = R.size();
            best = vector<int>(R.begin(), R.end());
        }
        return;
    }

    auto P_copy = P; 
    for (int v : P_copy) {
        set<int> newR = R;
        newR.insert(v);

        set<int> newP, newX;
        for (int u : P) {
            if (graph[v].count(u))
                newP.insert(u);
        }
        for (int u : X) {
            if (graph[v].count(u))
                newX.insert(u);
        }

        bron_kerbosch(newR, newP, newX);

        P.erase(v);
        X.insert(v);
    }
}

int main() {
    cin >> n >> m;
    graph.resize(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].insert(v);
        graph[v].insert(u);
    }

    set<int> P;
    for (int i = 1; i <= n; i++) P.insert(i);

    bron_kerbosch({}, P, {});

    cout << ans << "\n";
    for (int v : best) {
        cout << v << " ";
    }
    return 0;
}
