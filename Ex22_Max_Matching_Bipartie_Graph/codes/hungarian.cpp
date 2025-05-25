#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

const int MAXN = 10001;
const int MAXM = 10001;

int n, m;
vector<int> adj[MAXN];
int match[MAXM];
bool visited[MAXN];

bool dfs(int u) {
    if (visited[u]) return false;
    visited[u] = true;
    for (int v : adj[u]) {
        if (match[v] == -1 || dfs(match[v])) {
            match[v] = u;
            return true;
        }
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n >> m;
    for (int i = 1; i <= n; ++i) {
        int k;
        cin >> k;
        while (k--) {
            int v;
            cin >> v;
            adj[i].push_back(v);
        }
    }

    fill(match, match + MAXM, -1);
    int res = 0;

    for (int u = 1; u <= n; ++u) {
        fill(visited, visited + MAXN, false);
        if (dfs(u)) ++res;
    }

    cout << res;
    return 0;
}
