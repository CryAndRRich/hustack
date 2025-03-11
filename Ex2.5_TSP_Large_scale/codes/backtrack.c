#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

#define MAX_N 1000 

int n;
int best = INT_MAX;
int dist[MAX_N][MAX_N]; 
int bestPath[MAX_N];
int currPath[MAX_N];
bool visited[MAX_N];
int predict(int currCost) {
    int pred = 0;
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            int minEdge = INT_MAX;
            for (int j = 0; j < n; j++) {
                if (i != j && dist[i][j] < minEdge)
                    minEdge = dist[i][j];
            }
            if (minEdge != INT_MAX)
                pred += minEdge;
        }
    }
    return currCost + pred;
}

void backtrack(int curr, int count, int currCost) {
    if (count == n) {
        int total = currCost + dist[curr][currPath[0]];
        if (total < best) {
            best = total;
            for (int i = 0; i < n; i++)
                bestPath[i] = currPath[i];
        }
        return;
    }
    
    if (predict(currCost) >= best)
        return;
    
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            visited[i] = true;
            currPath[count] = i;
            backtrack(i, count + 1, currCost + dist[curr][i]);
            visited[i] = false;
        }
    }
}

int main(void) {
    scanf("%d", &n);
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &dist[i][j]);
    
    for (int start = 0; start < n; start++) {
        for (int i = 0; i < n; i++) 
            visited[i] = false;
        
        visited[start] = true;
        currPath[0] = start;
        backtrack(start, 1, 0);
    }
    
    printf("%d\n", n);
    for (int i = 0; i < n; i++)
        printf("%d ", bestPath[i] + 1);
    
    return 0;
}
