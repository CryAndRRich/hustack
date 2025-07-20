#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

string canonical_label(int n, const vector<pair<int,int>> &edges) {
    vector<vector<int>> adj(n, vector<int>(n, 0));
    for (const auto &e : edges) {
        int u = e.first - 1, v = e.second - 1;
        adj[u][v] = 1;
        adj[v][u] = 1;
    }
    
    vector<int> perm(n);
    for (int i = 0; i < n; i++) 
        perm[i] = i;
    
    string best;
    bool first = true;
    do {
        string rep;
        rep.reserve(n * n);
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                rep.push_back('0' + adj[perm[i]][perm[j]]);
            }
        }
        if (first || rep < best) {
            best = rep;
            first = false;
        }
    } while (next_permutation(perm.begin(), perm.end()));
    
    return best;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n1, m1;
    cin >> n1 >> m1;
    vector<pair<int,int>> edges1(m1);
    for (int i = 0; i < m1; i++){
        int u, v;
        cin >> u >> v;
        edges1[i] = {u, v};
    }
    
    int n2, m2;
    cin >> n2 >> m2;
    vector<pair<int,int>> edges2(m2);
    for (int i = 0; i < m2; i++){
        int u, v;
        cin >> u >> v;
        edges2[i] = {u, v};
    }
    
    if(n1 != n2){
        cout << 0;
        return 0;
    }
    
    string label1 = canonical_label(n1, edges1);
    string label2 = canonical_label(n2, edges2);
    
    cout << (label1 == label2 ? 1 : 0);
    return 0;
}
