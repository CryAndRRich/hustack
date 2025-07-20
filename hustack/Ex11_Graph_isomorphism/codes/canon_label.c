#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX 21      
#define MAX_STR 5000 

int n;             
int **adj;

char best[MAX_STR];
int best_init = 0;

void update_best(int perm[]) {
    char rep[MAX_STR];
    int idx = 0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            int val = adj[perm[i]][perm[j]];
            rep[idx++] = '0' + val;
        }
    }
    rep[idx] = '\0';
    if (!best_init || strcmp(rep, best) < 0) {
        strcpy(best, rep);
        best_init = 1;
    }
}

void swap(int *a, int *b) {
    int tmp = *a; 
    *a = *b; 
    *b = tmp;
}

void permute(int perm[], int l, int r) {
    if (l == r) {
        update_best(perm);
    } else {
        for (int i = l; i <= r; i++){
            swap(&perm[l], &perm[i]);
            permute(perm, l + 1, r);
            swap(&perm[l], &perm[i]);  
        }
    }
}

void canonical_label(int n_input, int m, int edges[][2], char *result) {
    n = n_input;
    adj = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++){
        adj[i] = (int *)calloc(n, sizeof(int));
    }
    
    for (int i = 0; i < m; i++){
        int u = edges[i][0] - 1;
        int v = edges[i][1] - 1;
        adj[u][v] = 1;
        adj[v][u] = 1;
    }
    
    int perm[MAX];
    for (int i = 0; i < n; i++)
        perm[i] = i;
    
    best_init = 0;
    best[0] = '\0';
    permute(perm, 0, n - 1);
    
    strcpy(result, best);
    
    for (int i = 0; i < n; i++){
        free(adj[i]);
    }
    free(adj);
}

int main(){
    int n1, m1;
    scanf("%d %d", &n1, &m1);

    int edges1[m1][2];
    for (int i = 0; i < m1; i++){
        if(scanf("%d %d", &edges1[i][0], &edges1[i][1]) != 2){
            return 1;
        }
    }
    
    int n2, m2;
    scanf("%d %d", &n2, &m2);

    int edges2[m2][2];
    for (int i = 0; i < m2; i++){
        if(scanf("%d %d", &edges2[i][0], &edges2[i][1]) != 2){
            return 1;
        }
    }
    
    if(n1 != n2){
        printf("%d", 0);
        return 0;
    }
    
    char label1[MAX_STR], label2[MAX_STR];
    canonical_label(n1, m1, edges1, label1);
    canonical_label(n2, m2, edges2, label2);
    
    if (strcmp(label1, label2) == 0)
        printf("%d", 1);
    else
        printf("%d", 0);
    
    return 0;
}
