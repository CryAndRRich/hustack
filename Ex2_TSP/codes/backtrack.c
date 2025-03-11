#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

int best = INT_MAX;
int predict(int curr, int n, int dist[21][21], bool visited[]) {
    int pred = 0;
    for (int i = 0; i < n; i += 1) {
        if (!visited[i]) {
            int minEdge = INT_MAX;
            for (int j = 0; j < n; j += 1) {
                if (i != j && dist[i][j] < minEdge)
                    minEdge = dist[i][j];
            }
            if (minEdge != INT_MAX)
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
    
    for (int i = 0; i < n; i += 1) {
        if (visited[i]) {
            continue;
        }
        
        visited[i] = true;
        backtrack(i, n, dist, count + 1, currCost + dist[curr][i], visited);
        visited[i] = false;
    }
}

int main(void) {
    int n;
    scanf("%d", &n);
    
    int dist[21][21];
    bool visited[21] = {0};
    
    for (int i = 0; i < n; i += 1)
        for (int j = 0; j < n; j += 1)
            scanf("%d", &dist[i][j]);
    
    visited[0] = true; 
    backtrack(0, n, dist, 1, 0, visited);
    printf("%d\n", best);
    
    return 0;
}
