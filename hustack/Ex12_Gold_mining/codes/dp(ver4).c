#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    int size;
    int *tree;  
} FenTree;

FenTree* createFenTree(int n) {
    FenTree *ft = (FenTree*)malloc(sizeof(FenTree));
    ft -> size = n;
    ft -> tree = (int*)malloc((n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) {
        ft -> tree[i] = 0;
    }
    return ft;
}

void fen_update(FenTree *ft, int index, int val) {
    while(index <= ft -> size) {
        if(val > ft -> tree[index])
            ft -> tree[index] = val;
        index += index & (-index);
    }
}

int fen_query_prefix(FenTree *ft, int index) {
    int res = 0;
    while(index > 0) {
        res = fmax(res, ft -> tree[index]);
        index -= index & (-index);
    }
    return res;
}

int fen_query_range(FenTree *ft, int L, int R, int *dp) {
    int res = 0;
    while(R >= L) {
        int start = R - (R & -R) + 1;
        if(start >= L) {
            res = fmax(res, ft -> tree[R]);
            R = start - 1;
        } else {
            res = fmax(res, dp[R]);
            R--;
        }
    }
    return res;
}

int main(void) {
    int n, L1, L2;
    scanf("%d %d %d", &n, &L1, &L2);
    
    int *a = (int*)malloc((n + 1) * sizeof(int));
    for (int i = 1; i <= n; i++) {
        scanf("%d", &a[i]);
    }
    
    int *dp = (int*)malloc((n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++)
        dp[i] = 0;
    int ans = 0;
    
    FenTree *ft = createFenTree(n);
    
    for (int i = 1; i <= n; i++) {
        int best_prev = 0;
        if(i - L1 >= 1) {
            int left = (i - L2 < 1) ? 1 : i - L2;
            int right = i - L1;
            best_prev = fen_query_range(ft, left, right, dp);
        }
        dp[i] = a[i] + best_prev;
        if(dp[i] > ans)
            ans = dp[i];
        fen_update(ft, i, dp[i]);
    }
    
    printf("%d", ans);
    
    free(a);
    free(dp);
    free(ft -> tree);
    free(ft);
    return 0;
}
