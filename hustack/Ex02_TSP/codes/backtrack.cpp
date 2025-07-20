#include <iostream>
#include <limits>
using namespace std;

int best = numeric_limits<int>::max();

int predict(int curr, int n, int dist[21][21], bool visited[]) {
    int pred = 0;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            int minEdge = numeric_limits<int>::max();
            for (int j = 0; j < n; j++) {
                if (i != j && dist[i][j] < minEdge)
                    minEdge = dist[i][j];
            }
            if (minEdge != numeric_limits<int>::max())
                pred += minEdge;
        }
    }
    return curr + pred;
}

void backtrack(int curr, int n, int dist[21][21], int count, int currCost, bool visited[]) {
    if (count == n) {
        int total = currCost + dist[curr][0];
        if (total < best)
            best = total;
        return;
    }
    
    if (predict(currCost, n, dist, visited) >= best)
        return;
    
    for (int i = 0; i < n; i++) {
        if (visited[i])
            continue;
        
        visited[i] = true;
        backtrack(i, n, dist, count + 1, currCost + dist[curr][i], visited);
        visited[i] = false;
    }
}

int main() {
    int n;
    cin >> n;
    
    int dist[21][21];
    bool visited[21] = {false};
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> dist[i][j];
    
    visited[0] = true; 
    backtrack(0, n, dist, 1, 0, visited);
    cout << best;
    
    return 0;
}
