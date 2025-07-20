#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

const int MAX = 21;

int n;
int n1, n2;
int m1, m2;
bool adj1[MAX][MAX];
bool adj2[MAX][MAX];
int deg1[MAX], deg2[MAX];
vector<int> order;
int mp[MAX];
bool visited[MAX];

bool cmp(int a, int b) {
    return deg1[a] > deg1[b];
}

bool backtrack(int pos) {
    if (pos == n)
        return true;
    int u = order[pos];
    for (int v = 0; v < n; v += 1) {
        if (!visited[v] && deg1[u] == deg2[v]) {
            mp[u] = v;
            visited[v] = true;
            bool valid = true;
            for (int i = 0; i < n; i += 1) {
                if (mp[i] != -1) {
                    if (adj1[u][i] != adj2[v][mp[i]]) {
                        valid = false;
                        break;
                    }
                }
            }
            if (valid && backtrack(pos + 1))
                return true;
            mp[u] = -1;
            visited[v] = false;
        }
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n1 >> m1;
    for (int i = 0; i < n1; i += 1) {
        for (int j = 0; j < n1; j += 1) {
            adj1[i][j] = false;
            adj2[i][j] = false;
        }
    }
    
    for (int i = 0; i < m1; i += 1) {
        int u, v;
        cin >> u >> v;
        adj1[u - 1][v - 1] = true;
        adj1[v - 1][u - 1] = true;
    }
    
    cin >> n2 >> m2;
    if (n1 != n2) {
        cout << 0;
        return 0;
    }
    n = n1;
    for (int i = 0; i < m2; i += 1) {
        int u, v;
        cin >> u >> v;
        adj2[u - 1][v - 1] = true;
        adj2[v - 1][u - 1] = true;
    }
    
    for (int i = 0; i < n; i += 1) {
        deg1[i] = 0;
        deg2[i] = 0;
        for (int j = 0; j < n; j += 1) {
            if (adj1[i][j])
                deg1[i] += 1;
            if (adj2[i][j])
                deg2[i] += 1;
        }
    }
    
    order.resize(n);
    for (int i = 0; i < n; i += 1)
        order[i] = i;
    sort(order.begin(), order.end(), cmp);
    
    for (int i = 0; i < n; i += 1) {
        mp[i] = -1;
        visited[i] = false;
    }
    
    bool is_iso = backtrack(0);
    cout << (is_iso ? 1 : 0);
    return 0;
}
