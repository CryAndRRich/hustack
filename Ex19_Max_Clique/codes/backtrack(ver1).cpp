#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int n, m;
vector<vector<bool>> graph;
int ans = -1;
vector<int> curr;
vector<int> best;

void backtrack(int size, const vector<int>& cands, int count) {
    if (size + count <= ans) return;

    for (int i = 0; i < count; i++) {
        int v = cands[i];
        curr[size] = v;

        if (size + 1 > ans) {
            ans = size + 1;
            copy(curr.begin(), curr.begin() + size + 1, best.begin());
        }

        vector<int> newCand;
        for (int j = i + 1; j < count; j++) {
            int u = cands[j];
            if (graph[v][u]) {
                newCand.push_back(u);
            }
        }

        backtrack(size + 1, newCand, newCand.size());
    }
}

int main() {
    cin >> n >> m;

    graph.assign(n, vector<bool>(n, false));
    curr.resize(n);
    best.resize(n);

    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u - 1][v - 1] = true;
        graph[v - 1][u - 1] = true;
    }

    vector<int> cands(n);
    for (int i = 0; i < n; i++) {
        cands[i] = i;
    }

    backtrack(0, cands, n);

    cout << ans << '\n';
    for (int i = 0; i < ans; i++) {
        cout << best[i] + 1 << ' ';
    }

    return 0;
}
