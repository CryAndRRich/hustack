#include <bits/stdc++.h>
using namespace std;

int n, m;
bitset<100> adj[100];
bitset<100> best_clique;
int max_size = 0;

int popcount(const bitset<100>& b){
    return b.count();
}

void bron(bitset<100> R, bitset<100> P, bitset<100> X){
    if (P.none() && X.none()) {
        int sz = popcount(R);
        if (sz > max_size) {
            max_size = sz;
            best_clique = R;
        }
        return;
    }
    bitset<100> PX = P | X;
    int u = -1;
    for (int i = 0; i < n; i++){
        if (PX.test(i)){ u = i; break; }
    }
    bitset<100> cands = P & (~adj[u]);
    for (int v = 0; v < n; v++){
        if (!cands.test(v)) {
            continue;
        }
        R.set(v);
        bron(R, P & adj[v], X & adj[v]);
        R.reset(v);
        P.reset(v);
        X.set(v);
    }
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    cin >> n >> m;
    for (int i = 0; i < m; i++){
        int u, v;
        cin >> u >> v;
        adj[u - 1].set(v - 1);
        adj[v - 1].set(u - 1);
    }
    bitset<100> P;
    for (int i = 0; i < n; i++) {
        P.set(i);
    }
    bron({}, P, {});

    cout << max_size << "\n";
    for (int i = 0; i < n; i++){
        if (best_clique.test(i)){
            cout << i + 1 << " ";
        }
    }
    return 0;
}
