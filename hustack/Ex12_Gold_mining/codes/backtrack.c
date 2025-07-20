#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int n, L1, L2;
int best = 0;
int *a;      
int *upper;

void upper_bound() {
    for (int i = n; i >= 1; i--) {
        int best_next = 0;
        for (int j = i + L1; j <= n && j <= i + L2; j++) {
            if (upper[j] > best_next)
                best_next = upper[j];
        }
        upper[i] = a[i] + best_next;
    }
}

void backtrack(int i, int curr_sum) {
    if (curr_sum > best)
        best = curr_sum;
    
    if (curr_sum + (upper[i] - a[i]) <= best)
        return;
    
    for (int j = i + L1; j <= n && j <= i + L2; j++) {
        backtrack(j, curr_sum + a[j]);
    }
}

int main(void) {
    scanf("%d %d %d", &n, &L1, &L2);
    
    a = (int*)malloc((n + 1) * sizeof(int));
    upper = (int*)malloc((n + 1) * sizeof(int));
    
    for (int i = 1; i <= n; i++) {
        scanf("%d", &a[i]);
        upper[i] = a[i];
    }
    
    upper_bound();
    
    for (int i = 1; i <= n; i++) {
        backtrack(i, a[i]);
    }
    
    printf("%d", best);
    
    free(a);
    free(upper);
    return 0;
}
