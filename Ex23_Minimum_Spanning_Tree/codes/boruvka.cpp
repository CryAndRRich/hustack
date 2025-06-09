#include <iostream>
#include <vector>
#include <tuple>
#include <algorithm>

using namespace std;

const int MAXN = 100005;

int parent[MAXN], rank_arr[MAXN];
int n, m;

int find(int u) {
    while (u != parent[u]) {
        u = parent[u];
    }
    return u;
}

bool union_set(int u, int v) {
    int ru = find(u);
    int rv = find(v);
    if (ru == rv) return false;
    if (rank_arr[ru] < rank_arr[rv]) {
        parent[ru] = rv;
    } else {
        parent[rv] = ru;
        if (rank_arr[ru] == rank_arr[rv]) rank_arr[ru]++;
    }
    return true;
}

int main() {
    cin >> n >> m;
    vector<tuple<int, int, int>> edges(m);
    for (int i = 0; i < m; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        edges[i] = {w, u, v};
    }

    for (int i = 1; i <= n; ++i) {
        parent[i] = i;
        rank_arr[i] = 0;
    }

    int total = 0;
    int num_comp = n;

    while (num_comp > 1) {
        vector<tuple<int, int, int>*> cheapest(n + 1, nullptr);

        for (auto& edge : edges) {
            int w = get<0>(edge), u = get<1>(edge), v = get<2>(edge);
            int ru = find(u);
            int rv = find(v);
            if (ru == rv) continue;

            if (!cheapest[ru] || get<0>(*cheapest[ru]) > w)
                cheapest[ru] = &edge;
            if (!cheapest[rv] || get<0>(*cheapest[rv]) > w)
                cheapest[rv] = &edge;
        }

        for (int i = 1; i <= n; ++i) {
            if (cheapest[i]) {
                int w = get<0>(*cheapest[i]);
                int u = get<1>(*cheapest[i]);
                int v = get<2>(*cheapest[i]);
                if (union_set(u, v)) {
                    total += w;
                    num_comp--;
                }
            }
        }
    }

    cout << total;
    return 0;
}
