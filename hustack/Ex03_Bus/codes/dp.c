#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int main(){
    int n, k;
    scanf("%d %d", &n, &k);
    int size = 2 * n + 1;

    int **dist = (int**)malloc(sizeof(int*) * size);
    for (int i = 0; i < size; i++){
        dist[i] = (int*)malloc(sizeof(int) * size);
        for (int j = 0; j < size; j++){
            scanf("%d", &dist[i][j]);
        }
    }
    
    long long **dp = (long long**)malloc(sizeof(long long*) * (n+1));
    for (int i = 0; i <= n; i++){
        dp[i] = (long long*)malloc(sizeof(long long) * (n+1));
        for (int j = 0; j <= n; j++){
            dp[i][j] = 1000000000000000000LL;
        }
    }
    dp[0][0] = 0;  
    
    int **pre_i = (int**)malloc(sizeof(int*) * (n+1));
    int **pre_j = (int**)malloc(sizeof(int*) * (n+1));
    int **move = (int**)malloc(sizeof(int*) * (n+1)); 
    for (int i = 0; i <= n; i++){
        pre_i[i] = (int*)malloc(sizeof(int) * (n+1));
        pre_j[i] = (int*)malloc(sizeof(int) * (n+1));
        move[i] = (int*)malloc(sizeof(int) * (n+1));
        for (int j = 0; j <= n; j++){
            pre_i[i][j] = -1;
            pre_j[i][j] = -1;
            move[i][j] = -1;
        }
    }
    
    for (int i = 0; i <= n; i++){
        for (int j = 0; j <= i; j++){
            if (dp[i][j] == 1000000000000000000LL) continue;
            int pos;
            if (i == j) {
                pos = (i == 0 ? 0 : n + j);
            } else {
                pos = i;
            }
            
            if (i < n && (i - j) < k){
                int nextPick = i + 1; 
                long long newCost = dp[i][j] + dist[pos][nextPick];
                if (newCost < dp[i+1][j]){
                    dp[i+1][j] = newCost;
                    pre_i[i+1][j] = i;
                    pre_j[i+1][j] = j;
                    move[i+1][j] = 0;
                }
            }
            
            if (i > j){
                int nextDrop = n + j + 1;
                long long newCost = dp[i][j] + dist[pos][nextDrop];
                if (newCost < dp[i][j+1]){
                    dp[i][j+1] = newCost;
                    pre_i[i][j+1] = i;
                    pre_j[i][j+1] = j;
                    move[i][j+1] = 1;
                }
            }
        }
    }
    
    long long totalCost = dp[n][n] + dist[2*n][0];
   
    int *path = (int*)malloc(sizeof(int) * 2 * n);
    int cur_i = n, cur_j = n;
    int ind = 2 * n - 1;
    while (!(cur_i == 0 && cur_j == 0)) {
        int m = move[cur_i][cur_j]; 
        if (m == -1) break; 
        if (m == 0) {
            path[ind] = cur_i;
        } else {
            path[ind] = n + cur_j;
        }
        ind--;
        int temp_i = pre_i[cur_i][cur_j];
        int temp_j = pre_j[cur_i][cur_j];
        cur_i = temp_i;
        cur_j = temp_j;
    }

    printf("%d\n", n);
    for (int i = 0; i < 2 * n; i++){
        printf("%d ", path[i]);
    }

    for (int i = 0; i < size; i++){
        free(dist[i]);
    }
    free(dist);
    
    for (int i = 0; i <= n; i++){
        free(dp[i]);
        free(pre_i[i]);
        free(pre_j[i]);
        free(move[i]);
    }
    free(dp);
    free(pre_i);
    free(pre_j);
    free(move);
    free(path);
    
    return 0;
}
