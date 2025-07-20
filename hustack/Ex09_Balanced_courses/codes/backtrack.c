#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int m, n;
int **P;       
int *pSize;    
int *pSizeSize; 

bool **B;        
int *x;          
int *load;      
int best = 1000000000; 

bool check(int v, int k) {
    for (int j = 1; j < k; j++) {
        if (B[j][k] && v == x[j])
            return false;
    }
    return true;
}

void solve(int k) {
    int predict = 0;
    for (int i = 1; i <= m; i++) {
        if (load[i] > predict)
            predict = load[i];
    }
    
    if (predict >= best)
        return;

    for (int i = 0; i < pSize[k]; i++) {
        int v = P[k][i];
        if (check(v, k)) {
            x[k] = v;
            load[v]++;
            if (k == n) {
                int maxi = 0;
                for (int j = 1; j <= m; j++) {
                    if (load[j] > maxi)
                        maxi = load[j];
                }
                if (maxi < best)
                    best = maxi;
            } else {
                if (load[v] < best)
                    solve(k + 1);
            }
            load[v]--;
        }
    }
}

int main(){
    scanf("%d %d", &m, &n);

    P = (int **)malloc((n + 1) * sizeof(int *));
    pSize = (int *)malloc((n + 1) * sizeof(int));
    pSizeSize = (int *)malloc((n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) {
        pSize[i] = 0;
        pSizeSize[i] = 10; 
        P[i] = (int *)malloc(pSizeSize[i] * sizeof(int));
    }
    
    for (int i = 1; i <= m; i++) {
        int count;
        scanf("%d", &count);
        for (int j = 0; j < count; j++) {
            int task;
            scanf("%d", &task);
            if (task >= 1 && task <= n) {
                if (pSize[task] == pSizeSize[task]) {
                    pSizeSize[task] *= 2;
                    P[task] = (int *)realloc(P[task], pSizeSize[task] * sizeof(int));
                }
                P[task][pSize[task]++] = i;
            }
        }
    }
    
    int k;
    scanf("%d", &k);
    
    B = (bool **)malloc((n + 1) * sizeof(bool *));
    for (int i = 0; i <= n; i++) {
        B[i] = (bool *)malloc((n + 1) * sizeof(bool));
        for (int j = 0; j <= n; j++)
            B[i][j] = false;
    }
    for (int i = 0; i < k; i++) {
        int a, b;
        scanf("%d %d", &a, &b);
        B[a][b] = true;
        B[b][a] = true;
    }
    
    x = (int *)malloc((n + 1) * sizeof(int));
    load = (int *)malloc((m + 1) * sizeof(int));
    
    for (int i = 0; i <= n; i++)
        x[i] = 0;
    for (int i = 0; i <= m; i++)
        load[i] = 0;
    
    solve(1);
    
    printf("%d\n", best);
    
    for (int i = 0; i <= n; i++) {
        free(P[i]);
    }
    free(P);
    free(pSize);
    free(pSizeSize);
    for (int i = 0; i <= n; i++) {
        free(B[i]);
    }
    free(B);
    free(x);
    free(load);
    
    return 0;
}
