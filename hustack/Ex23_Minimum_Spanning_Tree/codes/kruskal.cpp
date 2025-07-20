#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    int u, v, w;
};

bool cmp(const Edge &a, const Edge &b) {
    return a.w < b.w;
}

const int MAXN = 100005;
int parent[MAXN], rank_arr[MAXN];

int find(int u) {
    while (u != parent[u]) {
        u = parent[u];
    }
    return u;
}

bool union_sets(int u, int v) {
    int ru = find(u);
    int rv = find(v);
    if (ru == rv) return false;
    if (rank_arr[ru] < rank_arr[rv]) {
        parent[ru] = rv;
    } else {
        parent[rv] = ru;
        if (rank_arr[ru] == rank_arr[rv]) {
            rank_arr[ru]++;
        }
    }
    return true;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector<Edge> edges(m);
    for (int i = 0; i < m; ++i) {
        cin >> edges[i].u >> edges[i].v >> edges[i].w;
    }

    sort(edges.begin(), edges.end(), cmp);

    for (int i = 1; i <= n; ++i) parent[i] = i;

    long long total = 0;
    for (const Edge &e : edges) {
        if (union_sets(e.u, e.v)) {
            total += e.w;
        }
    }

    cout << total;
    return 0;
}
