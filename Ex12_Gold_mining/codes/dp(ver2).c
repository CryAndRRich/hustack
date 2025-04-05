#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n, L1, L2;
    scanf("%d %d %d", &n, &L1, &L2);
    
    int *a = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i += 1) {
        scanf("%d", &a[i]);
    }
    
    int *dp = (int*)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i += 1) {
        dp[i] = 0;
    }
    
    int ans = 0;
    int *dq = (int*)malloc(n * sizeof(int));

    int front = 0, back = -1;  
    for (int i = 0; i < n; i += 1) {
        while (front <= back && dq[front] < i - L2) {
            front += 1;
        }
        
        if (i - L1 >= 0) {
            int j = i - L1;
            while (front <= back && dp[dq[back]] <= dp[j]) {
                back -= 1;
            }
            back += 1;
            dq[back] = j;
        }
        
        int best_prev = (front <= back) ? dp[dq[front]] : 0;
        dp[i] = a[i] + best_prev;
        
        if (dp[i] > ans) {
            ans = dp[i];
        }
    }
    
    printf("%d", ans);
    
    free(a);
    free(dp);
    free(dq);
    return 0;
}
