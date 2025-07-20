#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int n, m;
bool **graph;
int ans = -1;
int *curr;
int *best;

void backtrack(int size, int cands[], int count) {
    if (size + count <= ans) {
        return;
    }
    for (int i = 0; i < count; i++) {
        int v = cands[i];
        curr[size] = v;
        if (size + 1 > ans) {
            ans = size + 1;
            for (int k = 0; k < size + 1; k++) {
                best[k] = curr[k];
            }
        }

        int *newCand = malloc(count * sizeof(int));
        int cnt = 0;
        for (int j = i + 1; j < count; j++) {
            int u = cands[j];
            if (graph[v][u]) {
                newCand[cnt++] = u;
            }
        }
        
        backtrack(size + 1, newCand, cnt);
        free(newCand);
    }
}

int main() {
    scanf("%d %d", &n, &m);
    
    graph = malloc(n * sizeof(bool *));
    for (int i = 0; i < n; i++) {
        graph[i] = calloc(n, sizeof(bool));
    }
    
    for (int i = 0; i < m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        graph[u - 1][v - 1] = true;
        graph[v - 1][u - 1] = true;
    }
    
    curr = malloc(n * sizeof(int));
    best = malloc(n * sizeof(int));
    
    int *cands = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        cands[i] = i;
    }
    
    backtrack(0, cands, n);
    
    printf("%d\n", ans);
    for (int i = 0; i < ans; i++) {
        printf("%d ", best[i] + 1);
    }
    
    free(cands);
    free(curr);
    free(best);
    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
    
    return 0;
}
