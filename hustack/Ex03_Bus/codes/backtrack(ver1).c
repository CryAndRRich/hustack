#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <string.h>

#define PATH_BUFFER_SIZE 10000

int best = INT_MAX;           
char bestPath[PATH_BUFFER_SIZE] = "";
char pathBuffer[PATH_BUFFER_SIZE] = "";     

int predict(int curr, int n, bool visited[], int **dist) {
    int pred = 0;
    for (int i = 1; i < 2 * n + 1; i += 1) {
        if (!visited[i]) {
            int minEdge = INT_MAX;
            for (int j = 1; j < 2 * n + 1; j += 1) {
                if (i != j && dist[i][j] < minEdge)
                    minEdge = dist[i][j];
            }
            if (minEdge != INT_MAX)
                pred += minEdge;
        }
    }
    return curr + pred;
}

void backtrack(int curr, int n, int k, int deliveredCount, int load, int currCost, 
               int **dist, bool visited[], int pathLen) {
    int size = 2 * n + 1;
    if (deliveredCount == n) {
        int total = currCost + dist[curr][0]; 
        if (total < best) {
            best = total;
            pathBuffer[pathLen] = '\0'; 
            strcpy(bestPath, pathBuffer);
        }
        return;
    }
    
    if (predict(currCost, n, visited, dist) >= best)
        return;
        
    if (load < k) {
        for (int i = 1; i <= n; i++) {
            if (!visited[i]) {
                visited[i] = true;
                int written = snprintf(pathBuffer + pathLen, PATH_BUFFER_SIZE - pathLen, "%d ", i);
                backtrack(i, n, k, deliveredCount, load + 1, currCost + dist[curr][i],
                          dist, visited, pathLen + written);
                pathBuffer[pathLen] = '\0';
                visited[i] = false;
            }
        }
    }
    
    for (int i = 1; i <= n; i++) {
        if (visited[i] && !visited[i + n]) {
            visited[i + n] = true;
            int written = snprintf(pathBuffer + pathLen, PATH_BUFFER_SIZE - pathLen, "%d ", i + n);
            backtrack(i + n, n, k, deliveredCount + 1, load - 1, currCost + dist[curr][i + n],
                      dist, visited, pathLen + written);
            pathBuffer[pathLen] = '\0';
            visited[i + n] = false;
        }
    }
}

int main() {
    int n, k;
    scanf("%d %d", &n, &k);
    
    int size = 2 * n + 1;
    int **dist = (int **)malloc(size * sizeof(int *));

    for (int i = 0; i < size; i++) {
        dist[i] = (int *)malloc(size * sizeof(int));
    }
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            scanf("%d", &dist[i][j]);
        }
    }
    
    bool visited[2005] = { false };
    visited[0] = true; 
    
    best = INT_MAX;
    pathBuffer[0] = '\0'; 
    
    backtrack(0, n, k, 0, 0, 0, dist, visited, 0);
    
    printf("%d\n", n);
    printf("%s\n", bestPath);
    
    for (int i = 0; i < size; i++) {
        free(dist[i]);
    }
    free(dist);
    
    return 0;
}
