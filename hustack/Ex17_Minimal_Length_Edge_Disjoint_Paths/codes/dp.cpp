#include <bits/stdc++.h>
using namespace std;

#define MAXN 30
#define MAXM 50

int n, m;
int eu[MAXM], ev[MAXM], ec[MAXM];
int head[MAXN + 1], nxt[MAXM];
int to[MAXM], eid[MAXM];

struct path { uint64_t mask; int cost; };
path *paths;
int path_count = 0;
int paths_cap = 0;
int vis[MAXN + 1];

static inline void add_path(uint64_t mask, int cost) {
    if (path_count >= paths_cap) {
        paths_cap = paths_cap ? paths_cap * 2 : 1024;
        paths = (path*)realloc(paths, paths_cap * sizeof(path));
    }
    paths[path_count].mask = mask;
    paths[path_count].cost = cost;
    path_count++;
}

void dp(int u, uint64_t mask, int cost) {
    if (u == n) {
        add_path(mask, cost);
        return;
    }
    for (int i = head[u]; i != -1; i = nxt[i]) {
        int v = to[i];
        int id = eid[i];
        uint64_t bit = (uint64_t)1 << id;
        if (mask & bit) continue;
        if (vis[v]) continue;
        vis[v] = 1;
        dp(v, mask | bit, cost + ec[id]);
        vis[v] = 0;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    cin >> n >> m;
    for (int i = 1; i <= n; i++) head[i] = -1;
    for (int i = 0; i < m; i++) {
        cin >> eu[i] >> ev[i] >> ec[i];
        to[i] = ev[i];
        eid[i] = i;
        nxt[i] = head[eu[i]];
        head[eu[i]] = i;
    }
    paths = nullptr;
    path_count = 0;
    vis[1] = 1;
    dp(1, 0ULL, 0);
    vis[1] = 0;
    if (path_count < 2) {
        cout << "NOT_FEASIBLE";
        return 0;
    }
    int best = INT_MAX;
    for (int i = 0; i < path_count; i++) {
        for (int j = i + 1; j < path_count; j++) {
            if ((paths[i].mask & paths[j].mask) == 0) {
                int s = paths[i].cost + paths[j].cost;
                if (s < best) best = s;
            }
        }
    }
    if (best == INT_MAX) {
        cout << "NOT_FEASIBLE";
    } else {
        cout << best;
    }
    return 0;
}