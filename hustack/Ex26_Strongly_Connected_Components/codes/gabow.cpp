#include <iostream>
#include <vector>
#include <stack>
#include <cstring>

using namespace std;

const int MAXN = 300005;
const int MAXM = 1000005;

int N, M;
vector<int> adj[MAXN];

int dfs_num[MAXN], dfs_clock = 0;
bool in_stack[MAXN];
stack<int> S, P;
int scc_count = 0;

void gabow(int u) {
    dfs_num[u] = ++dfs_clock;
    S.push(u);
    P.push(u);
    in_stack[u] = true;

    for (int v : adj[u]) {
        if (dfs_num[v] == 0) {
            gabow(v);
        } else if (in_stack[v]) {
            while (dfs_num[P.top()] > dfs_num[v]) {
                P.pop();
            }
        }
    }

    if (P.top() == u) {
        P.pop();
        int v;
        do {
            v = S.top(); S.pop();
            in_stack[v] = false;
        } while (v != u);
        scc_count++;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> N >> M;
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u - 1].push_back(v - 1);  
    }

    memset(dfs_num, 0, sizeof(dfs_num));
    memset(in_stack, false, sizeof(in_stack));

    for (int i = 0; i < N; ++i) {
        if (dfs_num[i] == 0)
            gabow(i);
    }

    cout << scc_count;
    return 0;
}
